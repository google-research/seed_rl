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

"""Tests for seed_rl.grpc.python.ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from concurrent import futures
import threading
import time
import uuid

from absl.testing import parameterized
import numpy as np
from seed_rl.grpc.python import ops
from six.moves import range
import tensorflow as tf

Some = collections.namedtuple('Some', 'a b')


class OpsTest(tf.test.TestCase, parameterized.TestCase):

  def get_unix_address(self):
    return 'unix:/tmp/%s' % uuid.uuid4()

  def test_simple(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()



  def test_simple_two_calls(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    self.assertAllEqual(44, client.foo(43))
    server.shutdown()

  def test_empty_input(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[])
    def foo():
      return 42

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, client.foo())
    server.shutdown()

  def test_empty_output(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):  
      return []

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual([], client.foo(42))
    server.shutdown()

  def test_no_output(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):  
      pass

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertIsNone(client.foo(42))
    server.shutdown()

  def test_large_tensor(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    t = tf.fill([10, 1024, 1024], 1)  # 40MB

    @tf.function(input_signature=[tf.TensorSpec([] + list(t.shape), tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(t + 1, client.foo(t))
    server.shutdown()

  def test_create_variable(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    state = [None]

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      if state[0] is None:
        with tf.device('/device:CPU:0'):
          state[0] = tf.Variable(42)
      with tf.device('/device:CPU:0'):
        return x + state[0]

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, state[0].read_value())
    self.assertAllEqual(43, client.foo(1))
    state[0].assign(0)
    self.assertAllEqual(1, client.foo(1))
    server.shutdown()

  def test_wait_for_server(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)

    def create_client():
      result = ops.Client(address)
      return result

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(create_client)

      time.sleep(2)
      server.start()
      self.assertAllEqual(43, f.result().foo(42))
      server.shutdown()

  def test_wait_for_server2(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)

    def create_and_send():
      client = ops.Client(address)
      self.assertAllEqual(43, client.foo(42))

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(create_and_send)

      time.sleep(2)
      server.start()
      f.result()
      server.shutdown()

  def test_upvalue(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    a = tf.constant(2)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x / a

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(21, client.foo(42))
    server.shutdown()

  def test_queue(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32], [()])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      if x.shape == (1,):
        q.enqueue_many([x])
      else:
        q.enqueue([x])
      return x

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    client.foo(42)
    self.assertAllEqual(42, q.dequeue())
    server.shutdown()

  def test_string(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def hello(x):
      return tf.strings.join([x, ' world'])

    server.bind(hello)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(b'hello world', client.hello('hello'))
    server.shutdown()

  def test_client_non_scalar_server_address(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'server_address must be a scalar'):
      ops.Client(['localhost:8000', 'localhost:8001'])

  def test_server_non_vector_server_addresses(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'server_addresses must be a vector'):
      ops.Server([['localhost:8000', 'localhost:8001']])

  def test_not_bound(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    with self.assertRaisesRegex(tf.errors.UnavailableError,
                                'No function was bound'):
      server.start()

  def test_binding_function_twice(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[])
    def foo():
      return 42

    server.bind(foo)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Function \'foo\' was bound twice.'):
      server.bind(foo)

  def test_starting_twice(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)

    server.start()
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Server is already started'):
      server.start()

  def test_invalid_number_of_arguments(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Expects 1 arguments, but 2 is provided'):
      client.foo([42, 43])
    server.shutdown()

  def test_invalid_type(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Expects arg\[0\] to be int32 but string is provided'):
      client.foo('foo')
    server.shutdown()

  def test_failing_function(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      tf.assert_equal(1, x)  # Will fail.
      return x

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'assertion failed'):
      client.foo(42)
    server.shutdown()

  def test_nests(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    signature = (tf.TensorSpec([], tf.int32, name='arg1'),
                 Some(
                     tf.TensorSpec([], tf.int32, name='arg2'), [
                         tf.TensorSpec([], tf.int32, name='arg3'),
                         tf.TensorSpec([], tf.int32, name='arg4')
                     ]))

    @tf.function(input_signature=signature)
    def foo(*args):
      return tf.nest.map_structure(lambda t: t + 1, args)

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    inputs = (1, Some(2, [3, 4]))
    expected_outputs = (2, Some(3, [4, 5]))
    outputs = client.foo(inputs)
    outputs = tf.nest.map_structure(lambda t: t.numpy(), outputs)
    tf.nest.assert_same_structure(expected_outputs, outputs)
    self.assertAllEqual(
        tf.nest.flatten(expected_outputs), tf.nest.flatten(outputs))
    server.shutdown()

  def test_call_after_shutdown(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    server.shutdown()
    with self.assertRaisesRegex(tf.errors.UnavailableError, 'server closed'):
      client.foo(42)

  def test_shutdown_while_in_call(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    is_waiting = threading.Event()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      tf.py_function(is_waiting.set, [], [])
      tf.py_function(time.sleep, [1], [])
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      is_waiting.wait()
      server.shutdown()
      with self.assertRaisesRegex(tf.errors.UnavailableError, 'server closed'):
        f.result()

  def test_shutdown_while_in_blocking_call(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      q.enqueue(x)
      q.enqueue(x)
      q.enqueue(x)
      return x

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      q.dequeue()  # Wait for function to be called.
      server.shutdown()
      try:
        f.result()
        # Non-deterministic if server manage to send CancelledError before
        # shutting down or not.
      except tf.errors.CancelledError:
        pass
      except tf.errors.UnavailableError:
        pass

  def test_deletion_while_in_blocking_call(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      q.enqueue(x)
      q.enqueue(x)
      q.enqueue(x)
      return x

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      q.dequeue()  # Wait for function to be called.
      del server
      try:
        f.result()
        # Non-deterministic if server manage to send CancelledError before
        # shutting down or not.
      except tf.errors.CancelledError:
        pass
      except tf.errors.UnavailableError:
        pass

  def test_call_after_shutdown_and_start(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])  # To test cancellation is reset.

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      q.enqueue(x)
      return x + 1

    server.bind(foo)
    server.start()
    server.shutdown()
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()

  def test_no_batching_when_output_rank0(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[
        tf.TensorSpec([2], tf.int32),
        tf.TensorSpec([2], tf.int32)
    ])
    def foo(unused_x, unused_y):
      return 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Expects arg\[0\] to have shape with 1 dimension\(s\), '
        r'but had shape \[\]'):
      client.foo(1, 1)

  def test_shutdown_waiting_for_full_batch(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      time.sleep(1)
      server.shutdown()
      with self.assertRaisesRegex(tf.errors.UnavailableError, 'server closed'):
        f.result()

  def test_two_clients(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    client2 = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
      f0 = executor.submit(client.foo, 42)
      f1 = executor.submit(client2.foo, 44)

      self.assertAllEqual(43, f0.result())
      self.assertAllEqual(45, f1.result())
    server.shutdown()

  def test_not_fully_specified_outputs(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def foo(x):
      if tf.equal(x[0], 0):
        return tf.zeros([])
      elif tf.equal(x[0], 1):
        return tf.zeros([2])
      else:
        return tf.zeros([1])

    server.bind(foo)
    server.start()

    def client():
      client = ops.Client(address)
      with self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          'Output must be at least rank 1 when batching is enabled'):
        client.foo(0)

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
      f1 = executor.submit(client)
      f2 = executor.submit(client)
      f1.result()
      f2.result()

    server.shutdown()

  def test_not_fully_specified_outputs2(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([1], tf.int32)])
    def foo(x):
      result, = tf.py_function(lambda x: x, [x], [tf.int32])
      result.set_shape([None])
      return result

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, client.foo(42))
    server.shutdown()

  def test_invalid_shape(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.int32)])
    def foo(x):
      return x

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Expects arg\[0\] to have shape with suffix \[3\], '
        r'but had shape \[3,4\]'):
      client.foo(tf.zeros([3, 4], tf.int32))  # Shape [3, 4], not [4, 3]

    server.shutdown()

  def test_stress_test(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([5], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    num_clients = 10
    num_calls = 100
    clients = [ops.Client(address) for _ in range(num_clients)]

    def do_calls(client):
      for i in range(num_calls):
        self.assertAllEqual(i + 1, client.foo(i))

    with futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
      fs = [executor.submit(do_calls, client) for client in clients]

      for i, f in enumerate(futures.as_completed(fs), 0):
        f.result()
        if i == num_clients // 2:
          # Shutdown after at least half the clients have completed. Not
          # possible to wait on all because the last batch may not be filled up
          # so it can't complete.
          try:
            server.shutdown()
          except tf.errors.UnavailableError:
            pass
          break

  def test_tpu(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    with tf.device('/device:CPU:0'):
      a = tf.Variable(1)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      with tf.device('/device:CPU:0'):
        b = a + 1
        c = x + 1
      return x + b, c

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    a, b = client.foo(42)
    self.assertAllEqual(44, a)
    self.assertAllEqual(43, b)
    server.shutdown()

  def test_tpu_tf_function_same_device(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    with tf.device('/device:CPU:0'):
      a = tf.Variable(1)

    with tf.device('/device:CPU:0'):

      @tf.function
      def get_a_plus_one():
        return a + 1

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      with tf.device('/device:CPU:0'):
        b = x + get_a_plus_one()
      return b + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    a = client.foo(1)
    self.assertAllEqual(4, a)
    server.shutdown()

  def test_bind_multiple_functions(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
      return x + 1

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32)
    ])
    def bar(x, y):
      return x * y

    server.bind(foo)
    server.bind(bar)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    self.assertAllEqual(100, client.bar(10, 10))
    server.shutdown()

  def test_variable_out_of_scope(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    def bind():
      a = tf.Variable(1)

      @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
      def foo(x):
        return x + a

      server.bind(foo)
    bind()

    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()

  def test_batch_auto_detection(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(2, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    client = ops.Client(address)
    t = tf.constant([1, 2])
    self.assertAllEqual(t + 1, client.foo(t))
    server.shutdown()

  def test_client_side_batching(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([4], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo)
    server.start()

    def client(x):
      client = ops.Client(address)
      return client.foo(x)

    # Both clients send batches of size 2 to the server, the server is expected
    # to process it as a batch of size 4.
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
      f1 = executor.submit(client, np.array([42, 43], np.int32))
      f2 = executor.submit(client, np.array([142, 143], np.int32))
      self.assertAllEqual(f1.result(), [43, 44])
      self.assertAllEqual(f2.result(), [143, 144])

    server.shutdown()


if __name__ == '__main__':
  tf.test.main()
