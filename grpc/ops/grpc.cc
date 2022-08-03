// Copyright 2019 The SEED Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "absl/container/flat_hash_map.h"
#include "grpc/service.grpc.pb.h"
#include "grpc/service.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

using grpc::ChannelCredentials;
using grpc::CreateCustomChannel;
using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;

REGISTER_RESOURCE_HANDLE_OP(GrpcServerResource);

constexpr int workers_thread_pools = 26;

REGISTER_OP("CreateGrpcServer")
    .Input("handle: resource")
    .Input("server_addresses: string")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a gRPC server which binds @tf.function to calls.
)doc");

REGISTER_OP("GrpcServerBind")
    .Input("handle: resource")
    .Input("captures: Tcaptures")
    .Attr("fn_name: string")
    .Attr("fn: func")
    .Attr("Tcaptures: list(type) >= 0")

    .Attr("input_shapes: list(shape)")
    .Attr("output_shapes: list(shape)")
    .Attr("output_specs: string")
    .Attr("first_bind: bool")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Binds a tf.function to a call.
)doc");

REGISTER_OP("GrpcServerStart")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Starts the server.
)doc");

REGISTER_OP("GrpcServerShutdown")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Shutdowns the server and cancels all requests.
)doc");

REGISTER_RESOURCE_HANDLE_OP(GrpcClientResource);

REGISTER_OP("CreateGrpcClient")
    .Input("handle: resource")
    .Input("server_address: string")
    .Output("method_output_signatures: string")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Creates a gRPC client and initializes the channel.

Returns the list of methods available on the server.
)doc");

REGISTER_OP("GrpcClientCall")
    .Input("handle: resource")
    .Input("input_list: Tinput_list")
    .Attr("fn_name: string")
    .Attr("Tinput_list: list(type) >= 0")
    .Attr("Toutput_list: list(type) >= 0")
    .Output("output_list: Toutput_list")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Calls the server.
)doc");

namespace {

class FnType {
 public:
  virtual bool operator()(
      ServerContext* server_ctx, gtl::ArraySlice<Tensor> args,
      std::function<void(Status, std::vector<Tensor>)> callback) = 0;

  virtual void Shutdown() = 0;

  virtual ~FnType() {}
};

class TensorHandler final {
 public:
  explicit TensorHandler() {}

  void Init(ServerContext* ctx, const seed_rl::InitRequest* request,
            seed_rl::InitResponse* response) {
    for (auto& output_specs : output_specs_list_) {
      auto* signature = response->add_method_output_signature();
      signature->set_name(output_specs.first);
      *signature->mutable_output_specs() =
          output_specs.second.SerializeAsString();
    }
  }

  void Call(ServerContext* ctx, const seed_rl::CallRequest* request,
            seed_rl::CallResponse* response, std::function<void()> callback) {
    auto outer_callback = [response, callback](Status status,
                                               std::vector<Tensor> rets) {
      if (status.ok()) {
        TensorProto tp;
        for (const Tensor& t : rets) {
          t.AsProtoTensorContent(&tp);
          response->add_tensor(tp.SerializeAsString());
        }
      } else {
        response->set_status_code(status.code());
        response->set_status_error_message(status.error_message());
      }
      callback();
    };

    TensorProto tp;
    Status status;
    std::vector<Tensor> args(request->tensor_size());
    for (int i = 0; i < request->tensor_size(); ++i) {

      if (!tp.ParseFromString(request->tensor(i))) {
        status =
            Status(error::Code::INVALID_ARGUMENT, "Cannot parse TensorProto.");
        outer_callback(status, {});
        return;
      }
      CHECK(args[i].FromProto(tp));
    }

    auto it = fns_.find(request->function());
    if (it == fns_.end()) {
      Status status =
          errors::Internal("Function ", request->function(), " not found");
      outer_callback(status, {});
    } else {
      auto& bucket = it->second;
      if ((*bucket.fn[bucket.call_counter_ % bucket.fn.size()])(
              ctx, args, outer_callback)) {
        bucket.call_counter_++;
      }
    }
  }

  Status Bind(const string& fn_name, tensorflow::StructuredValue& output_specs,
              std::unique_ptr<FnType> fn, bool first_bind) {
    if (first_bind && fns_.contains(fn_name)) {
      return errors::InvalidArgument("Function '", fn_name,
                                     "' was bound twice.");
    }
    auto& bucket = fns_[fn_name];
    if (bucket.fn.empty()) {
      output_specs_list_.push_back(std::make_pair(fn_name, output_specs));
    }
    bucket.fn.push_back(std::move(fn));
    return Status::OK();
  }

  void Shutdown() {
    for (auto& list : fns_) {
      for (auto& f : list.second.fn) {
        f->Shutdown();
      }
    }
  }

  bool is_bound() const { return !output_specs_list_.empty(); }

 private:
  struct FnBucket {
    std::vector<std::unique_ptr<FnType>> fn;
    int call_counter_ = 0;
  };

  absl::flat_hash_map<string, FnBucket> fns_;
  std::vector<std::pair<string, tensorflow::StructuredValue>>
      output_specs_list_;
};

struct ProcessorContext {
  std::unique_ptr<grpc::ServerCompletionQueue> cq;
};

class Tag {
 public:
  virtual void Proceed() = 0;
  virtual string name() const = 0;
  virtual ~Tag() {}
};

class InitializedServer {
 public:
  InitializedServer(TensorHandler* handler, int size)
      : handler(handler), size(size) {
    states = new ProcessorContext[size];
  }

  ~InitializedServer() { delete[] states; }

  void Shutdown() {
    for (int x = 0; x < size; x++) {
      void* untyped_tag;
      bool ok;
      while (true) {
        auto res = states[x].cq->AsyncNext(&untyped_tag, &ok,
                                           gpr_now(GPR_CLOCK_MONOTONIC));
        if (res == grpc::CompletionQueue::TIMEOUT) {
          break;
        } else {
          CHECK(res == grpc::CompletionQueue::GOT_EVENT);
          Tag* tag = static_cast<Tag*>(untyped_tag);
          delete tag;
        }
      }
    }
    for (int x = 0; x < size; x++) {
      states[x].cq->Shutdown();
    }
    handler->Shutdown();
  }

  TensorHandler* handler;
  ProcessorContext* states;
  std::unique_ptr<Server> server;
  ServerBuilder builder;
  seed_rl::TensorService::AsyncService service;
  CancellationManager c_mgr;
  std::unique_ptr<thread::ThreadPool> tp;
  int size;
  std::atomic<bool> is_shutdown{false};
};

class InitData : Tag {
 public:
  InitData(std::shared_ptr<InitializedServer> server, int cq_id)
      : server_(server), cq_id_(cq_id), responder_(&ctx_) {
    server->service.RequestInit(&ctx_, &request_, &responder_,
                                server_->states[cq_id].cq.get(),
                                server_->states[cq_id].cq.get(), this);
  }

  void Proceed() override {
    if (status_ == PROCESS) {
      new InitData(server_, cq_id_);
      server_->handler->Init(&ctx_, &request_, &response_);
      status_ = FINISH;
      responder_.Finish(response_, grpc::Status::OK, this);
    } else {
      GPR_ASSERT(status_ == FINISH);
      delete this;
    }
  }

  string name() const override { return "Init"; }

 private:
  std::shared_ptr<InitializedServer> server_;
  int cq_id_;
  ServerContext ctx_;
  seed_rl::InitRequest request_;
  seed_rl::InitResponse response_;
  grpc::ServerAsyncResponseWriter<seed_rl::InitResponse> responder_;
  enum CallStatus { PROCESS, FINISH };
  CallStatus status_{PROCESS};  // The current serving state.
};

class CallData : Tag {
 public:
  CallData(std::shared_ptr<InitializedServer> server, int cq_id)
      : server_(server), cq_id_(cq_id), responder_(&ctx_) {
    server_->service.RequestCall(&ctx_, &responder_,
                                 server_->states[cq_id].cq.get(),
                                 server_->states[cq_id].cq.get(), this);
  }

  void Proceed() override {
    if (status_ == PROCESS) {
      new CallData(server_, cq_id_);
      status_ = READ;
      responder_.Read(&request_, this);
    } else if (status_ == READ) {
      response_.Clear();
      server_->handler->Call(&ctx_, &request_, &response_, [this]() {
        if (!server_->is_shutdown) {
          status_ = WRITE;
          responder_.Write(response_, this);
          return;
        }
        delete this;
      });
    } else /** if (status_ == WRITE) **/ {
      status_ = READ;
      responder_.Read(&request_, this);
    }
  }

  string name() const override { return "CallData"; }

 private:
  std::shared_ptr<InitializedServer> server_;
  int cq_id_;
  ServerContext ctx_;
  seed_rl::CallRequest request_;
  seed_rl::CallResponse response_;
  grpc::ServerAsyncReaderWriter<seed_rl::CallResponse, seed_rl::CallRequest>
      responder_;
  enum CallStatus { PROCESS, READ, WRITE };
  CallStatus status_ = PROCESS;  // The current serving state.
};

class GrpcServerResource : public ResourceBase {
 public:
  GrpcServerResource(
      std::vector<std::pair<string, std::shared_ptr<grpc::ServerCredentials>>>
          ports)
      : ResourceBase(),
        ports_(ports),
        num_polling_threads_(workers_thread_pools),
        func_tp(new thread::ThreadPool(tensorflow::Env::Default(), "batched_fn",
                                       workers_thread_pools)) {}

  string DebugString() const override { return "gRPC Server"; }

  TensorHandler* tensor_handler() { return &handler_; }

  Status create_child_cancellation_manager(
      std::shared_ptr<CancellationManager>* child,
      std::shared_ptr<std::function<void()>>* deregister_fn) {
    *child = std::make_shared<CancellationManager>();
    auto childv = *child;
    CHECK(initialized_server_ != nullptr);

    auto& c_mgr = initialized_server_->c_mgr;
    CancellationToken token = c_mgr.get_cancellation_token();
    if (!c_mgr.RegisterCallback(token, [childv]() { childv->StartCancel(); })) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = std::make_shared<std::function<void()>>(
        [&c_mgr, childv, token]() { c_mgr.DeregisterCallback(token); });

    return Status::OK();
  }

  Status BuildAndStartSever() {
    if (!tensor_handler()->is_bound()) {
      return errors::Unavailable("No function was bound");
    }

    if (initialized_server_) {
      return errors::InvalidArgument("Server is already started");
    }

    initialized_server_ =
        std::make_shared<InitializedServer>(&handler_, num_polling_threads_);

    initialized_server_->builder.SetMaxReceiveMessageSize(
        std::numeric_limits<int32>::max());
    initialized_server_->builder.SetMaxSendMessageSize(
        std::numeric_limits<int32>::max());

    for (auto& t : ports_) {
      initialized_server_->builder.AddListeningPort(t.first, t.second);
    }

    for (int i = 0; i < num_polling_threads_; ++i) {
      initialized_server_->states[i].cq =
          initialized_server_->builder.AddCompletionQueue();
    }

    initialized_server_->builder.RegisterService(&initialized_server_->service);

    initialized_server_->server = initialized_server_->builder.BuildAndStart();
    initialized_server_->tp.reset(new thread::ThreadPool(
        tensorflow::Env::Default(), "cq_processor", num_polling_threads_));

    for (int i = 0; i < num_polling_threads_; ++i) {
      initialized_server_->tp->Schedule([this, i]() {
        auto& cq = initialized_server_->states[i].cq;
        new CallData(initialized_server_, i);
        new InitData(initialized_server_, i);
        void* untyped_tag;
        bool ok;
        while (cq->Next(&untyped_tag, &ok)) {
          Tag* tag = static_cast<Tag*>(untyped_tag);
          if (ok) {
            tag->Proceed();
          } else {
            delete tag;
          }
        }
      });
    }
    return Status::OK();
  }

  void ShutdownServer() {
    initialized_server_->is_shutdown = true;
    initialized_server_->server->Shutdown(std::chrono::system_clock::now());
    initialized_server_->Shutdown();
    initialized_server_->tp.reset();
    initialized_server_->c_mgr.StartCancel();
    while (initialized_server_.use_count() > 1) {
      usleep(10000);
    }
    initialized_server_.reset();
  }

  ~GrpcServerResource() override {
    if (initialized_server_) {
      ShutdownServer();
    }
  }

  std::vector<std::pair<string, std::shared_ptr<grpc::ServerCredentials>>>
      ports_;
  const int num_polling_threads_;
  std::shared_ptr<InitializedServer> initialized_server_;
  TensorHandler handler_;
  std::unique_ptr<thread::ThreadPool> func_tp;
};

REGISTER_RESOURCE_HANDLE_KERNEL(GrpcServerResource);

class CreateGrpcServerOp : public OpKernel {
 public:
  explicit CreateGrpcServerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto insecure_creds = grpc::InsecureServerCredentials();

    std::vector<std::pair<string, std::shared_ptr<grpc::ServerCredentials>>>
        ports;
    const Tensor* server_addresses_t;
    OP_REQUIRES_OK(ctx, ctx->input("server_addresses", &server_addresses_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(server_addresses_t->shape()),
                errors::InvalidArgument(
                    "server_addresses must be a vector, got shape: ",
                    server_addresses_t->shape().DebugString()));

    for (int i = 0; i < server_addresses_t->NumElements(); ++i) {
      string server_address = server_addresses_t->vec<tstring>()(i);
      auto creds = insecure_creds;
      ports.push_back({server_address, creds});
    }

    auto resource = new GrpcServerResource(ports);

    OP_REQUIRES_OK(ctx, CreateResource(ctx, HandleFromInput(ctx, 0), resource));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CreateGrpcServerOp);
};

REGISTER_KERNEL_BUILDER(Name("CreateGrpcServer").Device(DEVICE_CPU),
                        CreateGrpcServerOp);

Status verify_args(const DataTypeVector& expected_arg_types,
                   const std::vector<TensorShape>& expected_arg_shapes,
                   int batching_dims_count,
                   gtl::ArraySlice<Tensor> actual_args) {
  unsigned int num_expected_arguments = expected_arg_types.size();
  if (num_expected_arguments != actual_args.size()) {
    return errors::InvalidArgument("Expects ", num_expected_arguments,
                                   " arguments, but ", actual_args.size(),
                                   " is provided");
  }

  for (unsigned int i = 0; i < actual_args.size(); ++i) {
    if (expected_arg_shapes[i].dims() + batching_dims_count !=
        actual_args[i].shape().dims()) {
      return errors::InvalidArgument(
          "Expects arg[", i, "] to have shape with ",
          expected_arg_shapes[i].dims() + batching_dims_count,
          " dimension(s), but had shape ",
          actual_args[i].shape().DebugString());
    }
    if (!TensorShapeUtils::EndsWith(actual_args[i].shape(),
                                    expected_arg_shapes[i])) {
      return errors::InvalidArgument(
          "Expects arg[", i, "] to have shape with suffix ",
          expected_arg_shapes[i].DebugString(), ", but had shape ",
          actual_args[i].shape().DebugString());
    }

    if (expected_arg_types[i] != actual_args[i].dtype()) {
      return errors::InvalidArgument(
          "Expects arg[", i, "] to be ", DataTypeString(expected_arg_types[i]),
          " but ", DataTypeString(actual_args[i].dtype()), " is provided");
    }
  }

  return Status::OK();
}

// Checks that @actual_args conform to the expected types and shapes.
// Determines whether @actual_args are batched. Batched arguments have expected
// shapes prefixed with the batching dimension. If the arguments do not
// match expected types or shapes, returns an error status.
// Otherwise, writes the batch size in @arg_batch_size.
// If the arguments are not batched, writes 0 in @arg_batch_size.
Status GetArgBatchSize(const DataTypeVector& expected_arg_types,
                       const std::vector<TensorShape>& expected_arg_shapes,
                       gtl::ArraySlice<Tensor> actual_args,
                       int* arg_batch_size) {
  const bool batched =
      !actual_args.empty() && !expected_arg_shapes.empty() &&
      actual_args[0].shape().dims() == expected_arg_shapes[0].dims() + 1;
  const int batching_dims_count = batched ? 1 : 0;
  const Status status = verify_args(expected_arg_types, expected_arg_shapes,
                                    batching_dims_count, actual_args);
  if (!status.ok()) {
    return status;
  }

  if (!batched) {
    *arg_batch_size = 0;
    return Status::OK();
  }

  // Check that the batching dimension is the same for all arguments.
  const int expected_batch_size = actual_args[0].shape().dim_size(0);
  for (unsigned int i = 1; i < actual_args.size(); ++i) {
    if (actual_args[i].shape().dim_size(0) != expected_batch_size) {
      return errors::InvalidArgument("Expects arg[", i,
                                     "] to start with the batching dimension ",
                                     expected_batch_size, " but had shape ",
                                     actual_args[i].shape().DebugString());
    }
  }

  *arg_batch_size = expected_batch_size;
  return Status::OK();
}

class DynamicFn : public FnType {
 public:
  DynamicFn(FunctionLibraryRuntime* lib,
            FunctionLibraryRuntime::Handle f_handle,
            DataTypeVector&& input_types, std::vector<TensorShape> input_shapes,
            std::vector<Tensor>&& captures, GrpcServerResource* resource,
            thread::ThreadPool* tp, bool batched)
      : lib_(lib),
        f_handle_(f_handle),
        input_types_(std::move(input_types)),
        input_shapes_(std::move(input_shapes)),
        captures_(std::move(captures)),
        resource_(resource),
        batch_size_(batched ? input_shapes_[0].dim_size(0) : -1),
        tp_(tp),
        mu_(new mutex()) {
    if (batch_size_ != -1) {
      for (auto shape : input_shapes_) {
        shape.RemoveDim(0);
        arg_shapes_.push_back(shape);
      }
      current_computation_ = BuildEmptyComputation();
    }
  }

  ~DynamicFn() {
    Shutdown();
    delete current_computation_;
  }

  bool operator()(
      ServerContext* server_ctx, gtl::ArraySlice<Tensor> args,
      std::function<void(Status, std::vector<Tensor>)> callback) override {
    // Is this a direct call not involving server-side batching?
    // Exact match of the argument shapes means it was batched by the client.
    if (batch_size_ == -1 ||
        (!args.empty() && args[0].shape() == input_shapes_[0])) {
      return DirectCall(server_ctx, args, callback);
    }

    int arg_batch_size = 0;
    Status status =
        GetArgBatchSize(input_types_, arg_shapes_, args, &arg_batch_size);
    if (!status.ok()) {
      callback(status, {});
      return false;
    }
    const bool args_batched = arg_batch_size > 0;
    const int slice_count = args_batched ? arg_batch_size : 1;

    int64 index;
    Computation* computation = nullptr;
    {
      mutex_lock lock(*mu_);
      CHECK(current_computation_);
      index = next_index_;
      next_index_ += slice_count;
      computation = current_computation_;

      // Note: it can happen that incoming batches have different sizes,
      // in this case next_index_ can exceed batch_size_.

      CHECK_LE(next_index_, batch_size_) << "Learner-side batch size exceeded";
      if (next_index_ == batch_size_) {
        next_index_ = 0;
        if (!empty_computations_.empty()) {
          current_computation_ = empty_computations_.back();
          empty_computations_.pop_back();
        } else {
          current_computation_ = BuildEmptyComputation();
        }
      }
    }

    // Copy input tensors to the batched input tensors.
    if (!args_batched) {
      for (unsigned int i = 0; i < args.size(); ++i) {
        TF_CHECK_OK(batch_util::CopyElementToSlice(
            args[i], &computation->request[i], index));
      }
    } else {
      for (unsigned int i = 0; i < args.size(); ++i) {
        TF_CHECK_OK(batch_util::CopyContiguousSlices(
            args[i], 0, index, arg_batch_size, &computation->request[i]));
      }
    }

    // Populate the callback for the last slice in the batch.
    computation->callbacks[index + slice_count - 1] = callback;

    int num_ready = computation->num_ready += slice_count;
    if (num_ready == batch_size_) {
      // A full batch have been filled up, so the function should be executed.
      FunctionLibraryRuntime::Options f_opts;
      f_opts.create_rendezvous = true;
      std::shared_ptr<CancellationManager> c_mgr = nullptr;
      std::shared_ptr<std::function<void()>> deregister_fn = nullptr;
      auto status =
          resource_->create_child_cancellation_manager(&c_mgr, &deregister_fn);
      CHECK(status.ok());
      f_opts.cancellation_manager = c_mgr.get();
      auto f_callback = [this, deregister_fn, computation, c_mgr,
                         args_batched](Status f_status) {
        (*deregister_fn)();
        if (f_status.ok()) {
          for (unsigned int i = 0; i < computation->outputs.size(); ++i) {
            const auto& shape = computation->outputs[i].shape();
            if (shape.dims() <= 0) {
              f_status = errors::InvalidArgument(
                  "Output must be at least rank 1 when batching is enabled");
              break;
            }

            if (input_shapes_[0].dim_size(0) != shape.dim_size(0)) {
              f_status = errors::InvalidArgument(
                  "All outputs must have the same batch size "
                  "as the inputs when batching is enabled, expected: ",
                  input_shapes_[0].dim_size(0), " was: ", shape.dim_size(0));
              break;
            }
          }
        }

        // Parallel call all callbacks with their slice of outputs in.
        // Make sure computation is freed once callbacks are done.
        std::shared_ptr<Computation> done_computation;
        done_computation.reset(computation);
        int prev_batch_limit = -1;
        std::vector<std::pair<int, int>> batch_bounds;
        for (int j = 0; j < batch_size_; j++) {
          if (!computation->callbacks[j]) continue;
          batch_bounds.push_back(std::make_pair(prev_batch_limit + 1, j + 1));
          prev_batch_limit = j;
        }

        const int work_unit_size =
            (batch_bounds.size() + workers_thread_pools - 1) /
            workers_thread_pools;
        for (int j = 0; j < batch_bounds.size(); j += work_unit_size) {
          const int limit =
              std::min<int>(j + work_unit_size, batch_bounds.size());
          tp_->Schedule([j, done_computation, f_status, limit, batch_bounds,
                         args_batched]() {
            for (int x = j; x < limit; x++) {
              const int batch_start = batch_bounds[x].first;
              const int batch_limit = batch_bounds[x].second;
              std::vector<Tensor> rets;
              if (f_status.ok()) {
                rets.reserve(done_computation->outputs.size());
                // Pass the slice of the batched outputs to the return vector.
                for (unsigned int i = 0; i < done_computation->outputs.size();
                     ++i) {
                  if (args_batched) {
                    rets.push_back(done_computation->outputs[i].Slice(
                        batch_start, batch_limit));
                  } else {
                    rets.push_back(
                        done_computation->outputs[i].SubSlice(batch_start));
                  }
                }
              }
              // Callbacks are populated for the last slice in the batch.
              done_computation->callbacks[batch_limit - 1](f_status, rets);
            }
          });
        }
      };
      lib_->Run(f_opts, f_handle_, computation->request, &computation->outputs,
                f_callback);
      // Refill empty_computations_.
      Computation* refill_comp = BuildEmptyComputation();
      {
        mutex_lock lock(*mu_);
        empty_computations_.push_back(refill_comp);
      }
      return true;
    }
    return false;
  }

  void Shutdown() override {
    mutex_lock lock(*mu_);
    std::vector<Tensor> result;
    if (current_computation_) {
      for (auto& c : current_computation_->callbacks) {
        if (c) {
          c(errors::Cancelled("Server shutdown."), result);
        }
      }
      delete current_computation_;
      current_computation_ = BuildEmptyComputation();
    }
    for (auto c : empty_computations_) {
      delete c;
    }
    empty_computations_.clear();
  }

 private:
  // Represents one batched computation.
  struct Computation {
    std::vector<Tensor> request;
    std::vector<Tensor> outputs;
    std::vector<std::function<void(Status, std::vector<Tensor>)>> callbacks;
    std::atomic_int num_ready{0};
  };

  bool DirectCall(ServerContext* server_ctx, gtl::ArraySlice<Tensor> args,
                  std::function<void(Status, std::vector<Tensor>)> callback) {
    Status status = verify_args(input_types_, input_shapes_,
                                /*batching_dims_count=*/0, args);
    if (!status.ok()) {
      callback(status, {});
      return false;
    }
    FunctionLibraryRuntime::Options f_opts;
    f_opts.create_rendezvous = true;
    std::shared_ptr<CancellationManager> c_mgr;
    std::shared_ptr<std::function<void()>> deregister_fn;

    status =
        resource_->create_child_cancellation_manager(&c_mgr, &deregister_fn);

    if (!status.ok()) {
      callback(status, {});
      return false;
    }

    f_opts.cancellation_manager = c_mgr.get();

    std::vector<Tensor> full_args(args.begin(), args.end());
    for (auto& capture : captures_) {
      full_args.push_back(capture);
    }
    std::shared_ptr<std::vector<Tensor>> rets(new std::vector<Tensor>());
    lib_->Run(f_opts, f_handle_, full_args, rets.get(),
              [callback, rets, c_mgr](Status f_status) {
                callback(f_status, *rets);
              });
    return true;
  }

  Computation* BuildEmptyComputation() {
    Computation* c = new Computation();
    for (unsigned int i = 0; i < input_types_.size(); ++i) {
      c->request.emplace_back(input_types_[i], input_shapes_[i]);
    }
    for (const Tensor& t : captures_) {
      c->request.push_back(t);
    }
    c->callbacks.resize(batch_size_);
    return c;
  }

  FunctionLibraryRuntime* lib_;
  FunctionLibraryRuntime::Handle f_handle_;
  const DataTypeVector input_types_;
  const std::vector<TensorShape> input_shapes_;
  // input_shapes_ without batch dimension.
  std::vector<TensorShape> arg_shapes_;
  const std::vector<Tensor> captures_;
  GrpcServerResource* resource_;
  const int32 batch_size_;

  // HACK: A shared_ptr to make type copyable for std::function.
  thread::ThreadPool* tp_;
  std::shared_ptr<mutex> mu_;
  int64 next_index_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<Computation*> empty_computations_ ABSL_GUARDED_BY(mu_);
  Computation* current_computation_ ABSL_GUARDED_BY(mu_) = nullptr;
};

class GrpcServerBindOp : public OpKernel {
 public:
  explicit GrpcServerBindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fn_name", &fn_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fn", &fn_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_shapes", &input_shapes_));
    std::vector<PartialTensorShape> output_shapes;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes));
    string output_spec_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_specs", &output_spec_string));

    OP_REQUIRES(ctx, output_specs_.ParseFromString(output_spec_string),
                tensorflow::errors::InvalidArgument(
                    "Unable to parse StructuredValue output_spec string: ",
                    output_spec_string));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_bind", &first_bind_));
    batched_ = CanBatch(output_shapes);
  }

  void Compute(OpKernelContext* ctx) override {
    GrpcServerResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);

    OpInputList captures_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("captures", &captures_list));
    std::vector<Tensor> captures(captures_list.begin(), captures_list.end());

    FunctionLibraryRuntime* lib = ctx->function_library();
    const auto* flib_def = lib->GetFunctionLibraryDefinition();

    const FunctionDef* fdef = flib_def->Find(fn_.name());
    OP_REQUIRES(ctx, fdef != nullptr,
                errors::Internal("Failed to find function."));

    FunctionLibraryRuntime::Handle f_handle;
    FunctionLibraryRuntime::InstantiateOptions i_opts;
    i_opts.target = ctx->device()->name();
    i_opts.lib_def = flib_def;
    i_opts.is_multi_device_function = true;
    i_opts.create_kernels_eagerly = true;
    Device* cpu_device;
    OP_REQUIRES_OK(ctx, lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));
    int num_args = fdef->signature().input_arg_size();
    for (int i = 0; i < num_args - static_cast<int>(captures.size()); ++i) {
      i_opts.input_devices.push_back(cpu_device->name());
    }
    for (auto& captured : captures) {
      if (captured.dtype() == DT_RESOURCE) {
        const ResourceHandle& handle = captured.flat<ResourceHandle>()(0);
        i_opts.input_devices.push_back(handle.device());
      } else {
        i_opts.input_devices.push_back(cpu_device->name());
      }
    }

    // Make sure the outputs are on CPU so they can be copied.
    for (int i = 0; i < fdef->signature().output_arg_size(); ++i) {
      i_opts.output_devices.push_back(cpu_device->name());
    }

    OP_REQUIRES_OK(ctx, lib->Instantiate(fn_.name(), AttrSlice(&fn_.attr()),
                                         i_opts, &f_handle));
    OP_REQUIRES(
        ctx, num_args - captures.size() == input_shapes_.size(),
        errors::InvalidArgument(
            "Number of arguments is not the same size as input_shapes"));
    DataTypeVector input_types;
    for (unsigned int i = 0; i < input_shapes_.size(); ++i) {
      input_types.push_back(fdef->signature().input_arg(i).type());
    }
    DataTypeVector output_types;
    for (auto& output_arg : fdef->signature().output_arg()) {
      output_types.push_back(output_arg.type());
    }
    std::unique_ptr<FnType> func;
    func.reset(static_cast<FnType*>(new DynamicFn(
        lib, f_handle, std::move(input_types), input_shapes_,
        std::move(captures), resource, resource->func_tp.get(), batched_)));
    OP_REQUIRES_OK(
        ctx, resource->tensor_handler()->Bind(fn_name_, output_specs_,
                                              std::move(func), first_bind_));
  }

 private:
  bool CanBatch(const std::vector<PartialTensorShape>& output_shapes) {
    if (input_shapes_.empty()) {
      return false;
    }
    for (auto& shape : input_shapes_) {
      if (shape.dims() <= 0) {
        return false;
      }
      if (input_shapes_[0].dim_size(0) != shape.dim_size(0)) {
        return false;
      }
    }
    for (auto& shape : output_shapes) {
      if (shape.dims() == 0) {
        return false;
      }
      if (shape.dims() > 0 && shape.dim_size(0) != -1) {
        if (input_shapes_[0].dim_size(0) != shape.dim_size(0)) {
          return false;
        }
      }
    }
    return true;
  }

  string fn_name_;
  NameAttrList fn_;
  std::vector<TensorShape> input_shapes_;
  tensorflow::StructuredValue output_specs_;
  bool batched_;
  bool first_bind_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcServerBindOp);
};

REGISTER_KERNEL_BUILDER(Name("GrpcServerBind").Device(DEVICE_CPU),
                        GrpcServerBindOp);

class GrpcServerStartOp : public OpKernel {
 public:
  explicit GrpcServerStartOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    GrpcServerResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);
    OP_REQUIRES_OK(ctx, resource->BuildAndStartSever());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GrpcServerStartOp);
};

REGISTER_KERNEL_BUILDER(Name("GrpcServerStart").Device(DEVICE_CPU),
                        GrpcServerStartOp);

class GrpcServerShutdownOp : public OpKernel {
 public:
  explicit GrpcServerShutdownOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    GrpcServerResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);
    resource->ShutdownServer();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GrpcServerShutdownOp);
};

REGISTER_KERNEL_BUILDER(Name("GrpcServerShutdown").Device(DEVICE_CPU),
                        GrpcServerShutdownOp);

class GrpcClientResource : public ResourceBase {
 public:
  typedef grpc::ClientReaderWriter<seed_rl::CallRequest, seed_rl::CallResponse>
      ReaderWriter;

  GrpcClientResource() : ResourceBase() {}

  string DebugString() const override { return "gRPC client"; }

  grpc::Status Connect(const string& server_address,
                       std::shared_ptr<ChannelCredentials> creds,
                       const grpc::ChannelArguments& args,
                       std::vector<seed_rl::MethodOutputSignature>*
                           method_output_signature_list) {
    auto channel = CreateCustomChannel(server_address, creds, args);
    stub_ = seed_rl::TensorService::NewStub(channel);
    grpc::ClientContext init_ctx;
    init_ctx.set_wait_for_ready(true);
    seed_rl::InitRequest request;
    seed_rl::InitResponse response;
    grpc::Status status = stub_->Init(&init_ctx, request, &response);

    if (!status.ok()) {
      return status;
    }

    for (auto& method_output_signature : response.method_output_signature()) {
      method_output_signature_list->push_back(method_output_signature);
    }

    mutex_lock lock(call_mu_);
    stream_ = stub_->Call(&ctx_);
    return grpc::Status::OK;
  }

  Status Call(const seed_rl::CallRequest& request,
              seed_rl::CallResponse* response) {

    mutex_lock lock(call_mu_);
    if (!this->stream_->Write(request)) {
      return errors::Unavailable("Write failed, is the server closed?");
    }

    if (!this->stream_->Read(response)) {
      return errors::Unavailable("Read failed, is the server closed?");
    }

    return Status::OK();
  }

 private:
  std::unique_ptr<seed_rl::TensorService::Stub> stub_;
  grpc::ClientContext ctx_;
  std::unique_ptr<ReaderWriter> stream_ ABSL_GUARDED_BY(call_mu_);
  mutex call_mu_;
};

REGISTER_RESOURCE_HANDLE_KERNEL(GrpcClientResource);

class CreateGrpcClientOp : public OpKernel {
 public:
  explicit CreateGrpcClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

 private:
  void Compute(OpKernelContext* ctx) override {
    const Tensor* server_address_t;
    OP_REQUIRES_OK(ctx, ctx->input("server_address", &server_address_t));

    auto resource = new GrpcClientResource();
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(server_address_t->shape()),
        errors::InvalidArgument("server_address must be a scalar, got shape: ",
                                server_address_t->shape().DebugString()));
    string server_address = server_address_t->scalar<tstring>()();
    auto creds = grpc::InsecureChannelCredentials();
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
    args.SetMaxSendMessageSize(std::numeric_limits<int32>::max());
    args.SetCompressionAlgorithm(
        grpc_compression_algorithm::GRPC_COMPRESS_NONE);
    std::vector<seed_rl::MethodOutputSignature> method_output_signatures_list;
    Tensor* method_output_signatures_t;
    grpc::Status status = resource->Connect(server_address, creds, args,
                                            &method_output_signatures_list);
    OP_REQUIRES(ctx, status.ok(), errors::Unavailable(status.error_message()));
    auto method_output_signatures_shape =
        TensorShape({static_cast<int64>(method_output_signatures_list.size())});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, method_output_signatures_shape,
                                             &method_output_signatures_t));
    auto method_output_signatures = method_output_signatures_t->vec<tstring>();
    for (int i = 0; i < method_output_signatures_list.size(); ++i) {
      method_output_signatures(i) =
          method_output_signatures_list[i].SerializeAsString();
    }
    OP_REQUIRES_OK(ctx, CreateResource(ctx, HandleFromInput(ctx, 0), resource));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CreateGrpcClientOp);
};

REGISTER_KERNEL_BUILDER(Name("CreateGrpcClient").Device(DEVICE_CPU),
                        CreateGrpcClientOp);

class GrpcClientCallOp : public OpKernel {
 public:
  explicit GrpcClientCallOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fn_name", &fn_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    GrpcClientResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);
    OpInputList input_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("input_list", &input_list));
    seed_rl::CallRequest request;
    request.set_function(fn_name_);
    for (const Tensor& t : input_list) {
      TensorProto tp;
      t.AsProtoTensorContent(&tp);
      request.add_tensor(tp.SerializeAsString());
    }

    seed_rl::CallResponse response;
    {
      profiler::TraceMe trace_me([&]() {
        return absl::StrCat("Function ", fn_name_);
      });
      OP_REQUIRES_OK(ctx, resource->Call(request, &response));
    }

    OpOutputList output_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("output_list", &output_list));

    OP_REQUIRES(
        ctx, response.status_code() == error::OK,
        Status(static_cast<tensorflow::error::Code>(response.status_code()),
               response.status_error_message()));

    OP_REQUIRES(ctx, output_list.size() == response.tensor_size(),
                errors::InvalidArgument("Number of outputs was ",
                                        response.tensor_size(),
                                        " but expected ", output_list.size()));

    for (int i = 0; i < output_list.size(); ++i) {
      TensorProto tp;
      Tensor output_tensor;
      OP_REQUIRES(ctx, tp.ParseFromString(response.tensor(i)),
                  errors::Internal("Parsing of TensorProto failed."));
      OP_REQUIRES(ctx, output_tensor.FromProto(tp),
                  errors::Internal("Parsing of TensorProto failed."));
      ctx->set_output(i, output_tensor);
    }
  }

 private:
  string fn_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcClientCallOp);
};

REGISTER_KERNEL_BUILDER(Name("GrpcClientCall").Device(DEVICE_CPU),
                        GrpcClientCallOp);

}  // namespace
}  // namespace tensorflow
