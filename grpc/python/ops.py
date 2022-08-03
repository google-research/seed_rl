# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gRPC TensorFlow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import types

from seed_rl.grpc import service_pb2
from seed_rl.grpc.python.ops_wrapper import gen_grpc_ops
import tensorflow as tf


from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder



class Server(object):
  """A TensorFlow gRPC server."""

  def __init__(self, server_addresses):
    """Creates and starts the gRPC server.

    Args:
      server_addresses: A list of strings containing one or more server
        addresses.
    """
    if not tf.executing_eagerly():
      raise ValueError("Only eager mode is currently supported.")

    self._handle = gen_grpc_ops.grpc_server_resource_handle_op(
        shared_name=context.shared_name(None))
    # Delete the resource when this object is deleted.
    self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
        handle=self._handle, handle_device=context.context().device_name)
    gen_grpc_ops.create_grpc_server(self._handle, server_addresses)

    # References to tf.Variable's, etc. used in a tf.function to prevent them
    # from being deallocated.
    self._keep_alive = []

  def bind(self, fn):
    """Binds a tf.function to the server.

     If the first dimension of all
     arguments is equal (=N) then batching support is enabled, using N as
     a batch dimension. In such case when client does a call with a single
     element (batching dimention skipped), N independent client calls will be
     batched to construct an input for a single invocation of `fn`.
     If the signature of parameters provided by the client matches
     `input_signature`, `fn` is executed immediatelly without batching.

    Args:
      fn: The @tf.function wrapped function or a list of such functions, with
        `input_signature` set. When a list of functions is provided,
        they are called in a round-robin manner.

    Returns:
      A tf.Operation.
    """
    if not isinstance(fn, collections.Iterable):
      fn = [fn]

    for i, f in enumerate(fn):
      if f.input_signature is None:
        raise ValueError("tf.function must have input_signature set.")

      self._keep_alive.append(f.python_function)

      fn_name = f.__name__
      f = f.get_concrete_function()
      input_shapes = [
          t.shape for t in tf.nest.flatten(f.structured_input_signature)
      ]
      if f.structured_outputs is None:
        output_specs = None
      else:
        output_specs = tf.nest.map_structure(type_spec.type_spec_from_value,
                                             f.structured_outputs)
      encoder = nested_structure_coder.StructureCoder()
      output_specs_proto = encoder.encode_structure(output_specs)
      gen_grpc_ops.grpc_server_bind(
          handle=self._handle,
          captures=f.captured_inputs,
          fn_name=fn_name,
          fn=f,
          first_bind=(i == 0),
          input_shapes=input_shapes,
          output_shapes=tf.nest.flatten(f.output_shapes),
          output_specs=output_specs_proto.SerializeToString())

  def start(self):
    return gen_grpc_ops.grpc_server_start(handle=self._handle)

  def shutdown(self):
    return gen_grpc_ops.grpc_server_shutdown(handle=self._handle)


class Client(object):
  """A TensorFlow gRPC client."""

  # Disable pytype attribute checking.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, server_address):
    """Creates and starts the gRPC client.

    Args:
      server_address: A string containing the server address.
    """
    if not tf.executing_eagerly():
      raise ValueError("Only eager mode is currently supported.")

    self._handle = gen_grpc_ops.grpc_client_resource_handle_op(
        shared_name=context.shared_name(None))
    self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
        handle=self._handle, handle_device=context.context().device_name)
    method_signatures = gen_grpc_ops.create_grpc_client(self._handle,
                                                        server_address).numpy()
    m = service_pb2.MethodOutputSignature()
    v = struct_pb2.StructuredValue()
    for sig in method_signatures:
      assert m.ParseFromString(sig)
      decoder = nested_structure_coder.StructureCoder()
      assert v.ParseFromString(m.output_specs)
      decoded_output_specs = decoder.decode_proto(v)
      self._add_method(m.name, decoded_output_specs)

  def _add_method(self, name, output_specs):
    """Adds a method to the client."""
    flat_output_dtypes = [s.dtype for s in tf.nest.flatten(output_specs or [])]

    def call(self, *inputs):
      """Makes a call to the server."""

      flat_inputs = tf.nest.flatten(inputs)
      flat_outputs = gen_grpc_ops.grpc_client_call(
          fn_name=name,
          handle=self._handle,
          input_list=flat_inputs,
          Toutput_list=flat_output_dtypes)
      if output_specs is None:
        return None
      else:
        return tf.nest.pack_sequence_as(output_specs, flat_outputs)

    setattr(self, name, types.MethodType(call, self))
