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

import types

from seed_rl.grpc import service_pb2
from seed_rl.grpc.python.ops_wrapper import gen_grpc_ops
import tensorflow as tf

from google.protobuf import text_format


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

  def bind(self, fn, batched=False):
    """Binds a tf.function to the server.

    Args:
      fn: The @tf.function wrapped function. `input_signature` must be set.
      batched: If True, the function is batched and the first dimension is the
        batch dimension.

    Returns:
      A tf.Operation.
    """
    if fn.input_signature is None:
      raise ValueError("tf.function must have input_signature set.")


    self._keep_alive.append(fn.python_function)

    fn_name = fn.__name__
    fn = fn.get_concrete_function()
    input_shapes = [
        t.shape for t in tf.nest.flatten(fn.structured_input_signature)
    ]
    if fn.structured_outputs is None:
      output_specs = None
    else:
      output_specs = tf.nest.map_structure(type_spec.type_spec_from_value,
                                           fn.structured_outputs)
    encoder = nested_structure_coder.StructureCoder()
    output_specs_proto = encoder.encode_structure(output_specs)
    output_specs_string = text_format.MessageToString(output_specs_proto)
    return gen_grpc_ops.grpc_server_bind(
        handle=self._handle,
        captures=fn.captured_inputs,
        fn_name=fn_name,
        fn=fn,
        input_shapes=input_shapes,
        output_shapes=tf.nest.flatten(fn.output_shapes),
        output_specs=output_specs_string,
        batched=batched)

  def start(self):
    return gen_grpc_ops.grpc_server_start(handle=self._handle)

  def shutdown(self):
    return gen_grpc_ops.grpc_server_shutdown(handle=self._handle)


class Client(object):
  """A TensorFlow gRPC client."""

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
    method_signatures = [
        text_format.Parse(sig, service_pb2.MethodOutputSignature())
        for sig in method_signatures
    ]
    for sig in method_signatures:
      decoder = nested_structure_coder.StructureCoder()
      decoded_output_specs = decoder.decode_proto(sig.output_specs)
      self._add_method(sig.name, decoded_output_specs)

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
