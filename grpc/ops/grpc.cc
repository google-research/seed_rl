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
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/protobuf.h"
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
    .Attr("batched: bool")
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

class TensorServiceImpl final : public seed_rl::TensorService::Service {
 public:
  typedef std::function<Status(ServerContext*, gtl::ArraySlice<Tensor>,
                               std::vector<Tensor>*)>
      FnType;

  explicit TensorServiceImpl() {}

  grpc::Status Init(ServerContext* ctx, const seed_rl::InitRequest* request,
                    seed_rl::InitResponse* response) override {
    for (auto& output_specs : output_specs_list_) {
      auto* signature = response->add_method_output_signature();
      signature->set_name(output_specs.first);
      *signature->mutable_output_specs() =
          output_specs.second.SerializeAsString();
    }
    return grpc::Status::OK;
  }

  grpc::Status Call(ServerContext* ctx,
                    ServerReaderWriter<seed_rl::CallResponse,
                                       seed_rl::CallRequest>* stream) override {
    seed_rl::CallRequest request;
    TensorProto tp;
    while (stream->Read(&request)) {
      auto it = fns_.find(request.function());

      Status status;
      std::vector<Tensor> rets;
      if (it == fns_.end()) {
        status =
            errors::Internal("Function ", request.function(), " not found");
      } else {
        std::vector<Tensor> args(request.tensor_size());
        for (int i = 0; i < request.tensor_size(); ++i) {

          if (!tp.ParseFromString(request.tensor(i))) {
            status = Status(error::Code::INVALID_ARGUMENT,
                            "Cannot parse TensorProto.");
            break;
          }
          CHECK(args[i].FromProto(tp));
        }

        status = it->second(ctx, args, &rets);
      }

      seed_rl::CallResponse result;
      if (status.ok()) {
        TensorProto tp;
        for (const Tensor& t : rets) {
          t.AsProtoTensorContent(&tp);
          result.add_tensor(tp.SerializeAsString());
        }
      } else {
        result.set_status_code(status.code());
        result.set_status_error_message(status.error_message());
      }
      stream->Write(result);
    }
    return grpc::Status::OK;
  }

  Status Bind(const string& fn_name, tensorflow::StructuredValue& output_specs,
              FnType&& fn) {
    if (fns_.contains(fn_name)) {
      return errors::InvalidArgument("Function '", fn_name,
                                     "' was bound twice.");
    }

    output_specs_list_.push_back(std::make_pair(fn_name, output_specs));
    fns_[fn_name] = std::forward<FnType>(fn);

    return Status::OK();
  }

  bool is_bound() const { return !output_specs_list_.empty(); }

 private:
  absl::flat_hash_map<string, FnType> fns_;
  std::vector<std::pair<string, tensorflow::StructuredValue>>
      output_specs_list_;
};

class GrpcServerResource : public ResourceBase {
 public:
  GrpcServerResource()
      : ResourceBase(),
        builder_(absl::make_unique<ServerBuilder>()),
        service_(absl::make_unique<TensorServiceImpl>()) {}

  string DebugString() const override { return "gRPC Server"; }

  ServerBuilder* builder() { return builder_.get(); }

  TensorServiceImpl* service() { return service_.get(); }

  Status create_child_cancellation_manager(
      CancellationManager* child, std::function<void()>* deregister_fn) {
    CHECK(c_mgr_.get() != nullptr);

    CancellationToken token = c_mgr_->get_cancellation_token();
    if (!c_mgr_->RegisterCallback(token, [child]() { child->StartCancel(); })) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [this, token]() { c_mgr_->DeregisterCallback(token); };

    return Status::OK();
  }

  Status BuildAndStartSever() {
    if (!service_->is_bound()) {
      return errors::Unavailable("No function was bound");
    }

    if (server_ != nullptr) {
      return errors::InvalidArgument("Server is already started");
    }

    c_mgr_ = absl::make_unique<CancellationManager>();
    server_ = builder_->BuildAndStart();

    return Status::OK();
  }

  void ShutdownServer() {
    c_mgr_->StartCancel();
    server_->Shutdown(std::chrono::system_clock::now());
    c_mgr_.reset();
    server_.reset();
  }

  ~GrpcServerResource() override {
    if (server_ != nullptr) {
      ShutdownServer();
    }
  }

 private:
  std::unique_ptr<Server> server_;
  std::unique_ptr<ServerBuilder> builder_;
  std::unique_ptr<TensorServiceImpl> service_;
  std::unique_ptr<CancellationManager> c_mgr_;
};

REGISTER_RESOURCE_HANDLE_KERNEL(GrpcServerResource);

class CreateGrpcServerOp : public OpKernel {
 public:
  explicit CreateGrpcServerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto resource = new GrpcServerResource();

    auto insecure_creds = grpc::InsecureServerCredentials();

    const Tensor* server_addresses_t;
    OP_REQUIRES_OK(ctx, ctx->input("server_addresses", &server_addresses_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(server_addresses_t->shape()),
                errors::InvalidArgument(
                    "server_addresses must be a vector, got shape: ",
                    server_addresses_t->shape().DebugString()));

    for (int i = 0; i < server_addresses_t->NumElements(); ++i) {
      string server_address = server_addresses_t->vec<tstring>()(i);
      auto creds = insecure_creds;
      resource->builder()->AddListeningPort(server_address, creds);
    }
    resource->builder()->SetMaxReceiveMessageSize(
        std::numeric_limits<int32>::max());
    resource->builder()->SetMaxSendMessageSize(
        std::numeric_limits<int32>::max());
    resource->builder()->RegisterService(resource->service());

    OP_REQUIRES_OK(ctx, CreateResource(ctx, HandleFromInput(ctx, 0), resource));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CreateGrpcServerOp);
};

REGISTER_KERNEL_BUILDER(Name("CreateGrpcServer").Device(DEVICE_CPU),
                        CreateGrpcServerOp);

Status verify_args(const DataTypeVector& expected_arg_types,
                   const std::vector<TensorShape>& expected_arg_shapes,
                   gtl::ArraySlice<Tensor> actual_args) {
  unsigned int num_expected_arguments = expected_arg_types.size();
  if (num_expected_arguments != actual_args.size()) {
    return errors::InvalidArgument("Expects ", num_expected_arguments,
                                   " arguments, but ", actual_args.size(),
                                   " is provided");
  }

  for (unsigned int i = 0; i < actual_args.size(); ++i) {
    if (expected_arg_shapes[i] != actual_args[i].shape()) {
      return errors::InvalidArgument("Expects arg[", i, "] to have shape ",
                                     expected_arg_shapes[i].DebugString(),
                                     " but had shape ",
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

class BatchedFn {
 public:
  BatchedFn(FunctionLibraryRuntime* lib,
            FunctionLibraryRuntime::Handle f_handle,
            DataTypeVector&& input_types,
            std::vector<TensorShape>&& input_shapes,
            std::vector<Tensor>&& captures, GrpcServerResource* resource)
      : lib_(lib),
        f_handle_(f_handle),
        input_types_(std::move(input_types)),
        input_shapes_(std::move(input_shapes)),
        captures_(std::move(captures)),
        resource_(resource),
        batch_size_(input_shapes_[0].dim_size(0)),
        mu_(new mutex()) {
    for (auto shape : input_shapes_) {
      shape.RemoveDim(0);
      arg_shapes_.push_back(shape);
    }
  }

  Status operator()(ServerContext* server_ctx, gtl::ArraySlice<Tensor> args,
                    std::vector<Tensor>* rets) {
    TF_RETURN_IF_ERROR(verify_args(input_types_, arg_shapes_, args));

    int64 index;
    std::shared_ptr<Computation> computation;
    {
      mutex_lock lock(*mu_);
      index = next_index_++;

      if (index == 0) {
        CHECK(current_computation_ == nullptr);
        current_computation_ = std::make_shared<Computation>();
        for (unsigned int i = 0; i < input_types_.size(); ++i) {
          current_computation_->request.emplace_back(input_types_[i],
                                                     input_shapes_[i]);
        }
        for (const Tensor& t : captures_) {
          current_computation_->request.push_back(t);
        }
      }

      computation = current_computation_;

      if (index == batch_size_ - 1) {
        next_index_ = 0;
        current_computation_.reset();
      }
    }

    // Copy input tensors to the batched input tensors.
    for (unsigned int i = 0; i < args.size(); ++i) {
      TF_CHECK_OK(batch_util::CopyElementToSlice(
          args[i], &computation->request[i], index));
    }

    int num_ready = ++computation->num_ready;
    if (num_ready == batch_size_) {
      // A full batch have been filled up, so the function should be executed.
      FunctionLibraryRuntime::Options f_opts;
      f_opts.create_rendezvous = true;
      CancellationManager c_mgr;
      std::function<void()> deregister_fn;
      auto status =
          resource_->create_child_cancellation_manager(&c_mgr, &deregister_fn);
      f_opts.cancellation_manager = &c_mgr;
      auto done_callback = [computation](Status f_status) {
        computation->f_status.Update(f_status);
        computation->f_done.Notify();
      };
      if (status.ok()) {
        lib_->Run(f_opts, f_handle_, computation->request,
                  &computation->outputs, done_callback);
        computation->f_done.WaitForNotification();
        deregister_fn();
      } else {
        done_callback(status);
      }
    }

    // Wait for the function to run/finish.
    while (!WaitForNotificationWithTimeout(&computation->f_done,
                                           50000 /* 50 ms */)) {
      if (server_ctx->IsCancelled()) {
        break;
      }
    }

    // Save status because it's going to be freed before return.
    Status status;
    if (server_ctx->IsCancelled()) {
      // Exit if the server was cancelled.
      status = errors::Cancelled("Call was cancelled.");
    } else {
      status = computation->f_status;
    }

    if (status.ok()) {
      // Pass the slice of the batched outputs to the return vector.
      rets->resize(computation->outputs.size());
      for (unsigned int i = 0; i < computation->outputs.size(); ++i) {
        const auto& shape = computation->outputs[i].shape();
        if (shape.dims() <= 0) {
          status = errors::InvalidArgument(
              "Output must be at least rank 1 when batched=True");
          break;
        }

        if (input_shapes_[0].dim_size(0) != shape.dim_size(0)) {
          status = errors::InvalidArgument(
              "All outputs must have the same batch size "
              "as the inputs when batched=True, expected: ",
              input_shapes_[0].dim_size(0), " was: ", shape.dim_size(0));
          break;
        }

        (*rets)[i] = computation->outputs[i].SubSlice(index);
      }
    }

    return status;
  }

 private:
  // Represents one batched computation.
  struct Computation {
    std::vector<Tensor> request;
    std::vector<Tensor> outputs;
    Notification f_done;
    Status f_status;
    std::atomic_int num_ready{0};
  };

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
  std::shared_ptr<mutex> mu_;
  int64 next_index_ GUARDED_BY(mu_) = 0;
  std::shared_ptr<Computation> current_computation_ GUARDED_BY(mu_);
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

    OP_REQUIRES_OK(ctx, ctx->GetAttr("batched", &batched_));

    if (batched_) {
      OP_REQUIRES(
          ctx, !input_shapes_.empty(),
          errors::InvalidArgument(
              "Function must have at least one input when batched=True"));

      for (auto& shape : input_shapes_) {
        OP_REQUIRES(
            ctx, shape.dims() > 0,
            errors::InvalidArgument(
                "All inputs must at least be rank 1 when batched=True"));
        OP_REQUIRES(
            ctx, input_shapes_[0].dim_size(0) == shape.dim_size(0),
            errors::InvalidArgument("All inputs must have the same first "
                                    "dimension when batched=True"));
      }

      for (auto& shape : output_shapes) {
        OP_REQUIRES(
            ctx, shape.dims() == -1 || shape.dims() > 0,
            errors::InvalidArgument("All outputs must at least be rank 1 when "
                                    "batched=True but rank was: ",
                                    shape.dims()));
        if (shape.dims() > 0 && shape.dim_size(0) != -1) {
          OP_REQUIRES(
              ctx, input_shapes_[0].dim_size(0) == shape.dim_size(0),
              errors::InvalidArgument(
                  "All outputs must have the same batch size "
                  "as the inputs when batched=True, expected: ",
                  input_shapes_[0].dim_size(0), " was: ", shape.dim_size(0)));
        }
      }
    }
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
    if (batched_) {
      BatchedFn batched_f(lib, f_handle, std::move(input_types),
                          std::move(input_shapes_), std::move(captures),
                          resource);
      OP_REQUIRES_OK(
          ctx, resource->service()->Bind(fn_name_, output_specs_, batched_f));
    } else {
      auto input_shapes = std::move(input_shapes_);
      auto fn = [resource, captures, lib, f_handle, input_types, input_shapes,
                 output_types](ServerContext* server_ctx,
                               gtl::ArraySlice<Tensor> args,
                               std::vector<Tensor>* rets) {
        TF_RETURN_IF_ERROR(verify_args(input_types, input_shapes, args));
        FunctionLibraryRuntime::Options f_opts;
        f_opts.create_rendezvous = true;

        CancellationManager c_mgr;
        std::function<void()> deregister_fn;

        auto status =
            resource->create_child_cancellation_manager(&c_mgr, &deregister_fn);

        f_opts.cancellation_manager = &c_mgr;

        if (status.ok()) {
          std::vector<Tensor> full_args(args.begin(), args.end());
          for (auto& capture : captures) {
            full_args.push_back(capture);
          }
          Notification f_done;
          lib->Run(f_opts, f_handle, full_args, rets,
                   [&f_done, &status](Status f_status) {
                     status.Update(f_status);
                     f_done.Notify();
                   });
          f_done.WaitForNotification();
          deregister_fn();
        }

        return status;
      };
      OP_REQUIRES_OK(ctx,
                     resource->service()->Bind(fn_name_, output_specs_, fn));
    }
  }

 private:
  string fn_name_;
  NameAttrList fn_;
  std::vector<TensorShape> input_shapes_;
  tensorflow::StructuredValue output_specs_;
  bool batched_;

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
  std::unique_ptr<ReaderWriter> stream_ GUARDED_BY(call_mu_);
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
    OP_REQUIRES_OK(ctx, resource->Call(request, &response));

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
