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
import threading
import time
import uuid
from absl.testing import parameterized
from concurrent import futures
from seed_rl.grpc.python import ops

from six.moves import range

import tensorflow as tf


Some = collections.namedtuple('Some', 'a b')


class OpsTest(tf.test.TestCase, parameterized.TestCase):

  def get_unix_address(self):
    return 'unix:/tmp/%s' % uuid.uuid4()

  @parameterized.parameters(([], False), ([1], True))
  def test_simple(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()



  @parameterized.parameters(([], False), ([1], True))
  def test_simple_two_calls(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
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

    # Empty input is only allowed when batched=False.
    server.bind(foo, batched=False)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, client.foo())
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_empty_output(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):  
      return []

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual([], client.foo(42))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_no_output(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):  
      pass

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertIsNone(client.foo(42))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_large_tensor(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    t = tf.fill([10, 1024, 1024], 1)  # 40MB

    @tf.function(input_signature=[tf.TensorSpec(dim + list(t.shape), tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(t + 1, client.foo(t))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_create_variable(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    state = [None]

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      if state[0] is None:
        with tf.device('/device:CPU:0'):
          state[0] = tf.Variable(42)
      with tf.device('/device:CPU:0'):
        return x + state[0]

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, state[0].read_value())
    self.assertAllEqual(43, client.foo(1))
    state[0].assign(0)
    self.assertAllEqual(1, client.foo(1))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_wait_for_server(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)

    def create_client():
      result = ops.Client(address)
      return result

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(create_client)

      time.sleep(2)
      server.start()
      self.assertAllEqual(43, f.result().foo(42))
      server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_wait_for_server2(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)

    def create_and_send():
      client = ops.Client(address)
      self.assertAllEqual(43, client.foo(42))

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(create_and_send)

      time.sleep(2)
      server.start()
      f.result()
      server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_upvalue(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    a = tf.constant(2)

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x / a

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(21, client.foo(42))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_queue(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32], [()])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      if x.shape == (1,):
        q.enqueue_many([x])
      else:
        q.enqueue([x])
      return x

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    client.foo(42)
    self.assertAllEqual(42, q.dequeue())
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_string(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.string)])
    def hello(x):
      return tf.strings.join([x, ' world'])

    server.bind(hello, batched=batched)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(b'hello world', client.hello('hello'))
    server.shutdown()

  def test_client_non_scalar_server_address(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'server_address must be a scalar'):
      ops.Client(['localhost:8000', 'localhost:8001'])

  def test_server_non_vector_server_addresses(self):
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'server_addresses must be a vector'):
      ops.Server([['localhost:8000', 'localhost:8001']])

  def test_not_bound(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    with self.assertRaisesRegexp(tf.errors.UnavailableError,
                                 'No function was bound'):
      server.start()

  def test_binding_function_twice(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[])
    def foo():
      return 42

    server.bind(foo)
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
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
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Server is already started'):
      server.start()

  @parameterized.parameters(([], False), ([1], True))
  def test_invalid_number_of_arguments(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Expects 1 arguments, but 2 is provided'):
      client.foo([42, 43])
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_invalid_type(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        r'Expects arg\[0\] to be int32 but string is provided'):
      client.foo('foo')
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_failing_function(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      tf.assert_equal(1, x)  # Will fail.
      return x

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'assertion failed'):
      client.foo(42)
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_nests(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    signature = (tf.TensorSpec(dim, tf.int32, name='arg1'),
                 Some(
                     tf.TensorSpec(dim, tf.int32, name='arg2'), [
                         tf.TensorSpec(dim, tf.int32, name='arg3'),
                         tf.TensorSpec(dim, tf.int32, name='arg4')
                     ]))

    @tf.function(input_signature=signature)
    def foo(*args):
      return tf.nest.map_structure(lambda t: t + 1, args)

    server.bind(foo, batched=batched)
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

  @parameterized.parameters(([], False), ([1], True))
  def test_call_after_shutdown(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    server.shutdown()
    with self.assertRaisesRegexp(tf.errors.UnavailableError, 'server closed'):
      client.foo(42)

  @parameterized.parameters(([], False), ([1], True))
  def test_shutdown_while_in_call(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    is_waiting = threading.Event()

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      tf.py_function(is_waiting.set, [], [])
      tf.py_function(time.sleep, [1], [])
      return x + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      is_waiting.wait()
      server.shutdown()
      with self.assertRaisesRegexp(tf.errors.UnavailableError, 'server closed'):
        f.result()

  @parameterized.parameters(([], False), ([1], True))
  def test_shutdown_while_in_blocking_call(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      q.enqueue(x)
      q.enqueue(x)
      q.enqueue(x)
      return x

    server.bind(foo, batched=batched)
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

  @parameterized.parameters(([], False), ([1], True))
  def test_deletion_while_in_blocking_call(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      q.enqueue(x)
      q.enqueue(x)
      q.enqueue(x)
      return x

    server.bind(foo, batched=batched)
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

  @parameterized.parameters(([], False), ([1], True))
  def test_call_after_shutdown_and_start(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    q = tf.queue.FIFOQueue(1, [tf.int32])  # To test cancellation is reset.

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      q.enqueue(x)
      return x + 1

    server.bind(foo, batched=batched)
    server.start()
    server.shutdown()
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()

  def test_batched_first_dimension_must_match(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[
        tf.TensorSpec([1], tf.int32),
        tf.TensorSpec([2], tf.int32)
    ])
    def foo(x, y):
      return x, y

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'All inputs must have the same first dimension when batched=True'):
      server.bind(foo, batched=True)

  def test_batched_inputs_at_least_rank1(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[
        tf.TensorSpec([1], tf.int32),
        tf.TensorSpec([], tf.int32)
    ])
    def foo(x, y):
      return x, y

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'All inputs must at least be rank 1 when batched=True'):
      server.bind(foo, batched=True)

  def test_batched_outputs_at_least_rank1(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[
        tf.TensorSpec([1], tf.int32),
        tf.TensorSpec([1], tf.int32)
    ])
    def foo(unused_x, unused_y):
      return 1

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'All outputs must at least be rank 1 when batched=True'):
      server.bind(foo, batched=True)

  def test_batched_at_least_one_input(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[])
    def foo():
      return 1

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'Function must have at least one input when batched=True'):
      server.bind(foo, batched=True)

  def test_batched_output_is_batched(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([1], tf.int32)])
    def foo(unused_x):
      return tf.zeros([3])

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'All outputs must have the same batch size as the inputs.'):
      server.bind(foo, batched=True)

  def test_shutdown_waiting_for_full_batch(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=True)
    server.start()

    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      f = executor.submit(client.foo, 42)
      time.sleep(1)
      server.shutdown()
      with self.assertRaisesRegexp(tf.errors.UnavailableError, 'server closed'):
        f.result()

  @parameterized.parameters(([], False), ([1], True), ([2], True))
  def test_two_clients(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=batched)
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

    @tf.function(input_signature=[tf.TensorSpec([1], tf.int32)])
    def foo(x):
      if tf.equal(x[0], 0):
        return tf.zeros([])
      elif tf.equal(x[0], 1):
        return tf.zeros([2])
      else:
        return tf.zeros([1])

    server.bind(foo, batched=True)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(0, client.foo(42))

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'Output must be at least rank 1 when batched=True'):
      client.foo(0)

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        'All outputs must have the same batch size as '
        'the inputs when batched=True, expected: 1 was: 2'):
      client.foo(1)

    server.shutdown()

  def test_not_fully_specified_outputs2(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([1], tf.int32)])
    def foo(x):
      result, = tf.py_function(lambda x: x, [x], [tf.int32])
      result.set_shape([None])
      return result

    server.bind(foo, batched=True)
    server.start()

    client = ops.Client(address)
    self.assertAllEqual(42, client.foo(42))
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_invalid_shape(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim + [4, 3], tf.int32)])
    def foo(x):
      return x

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError,
        r'Expects arg\[0\] to have shape \[4,3\] but had shape \[3,4\]'):
      client.foo(tf.zeros([3, 4], tf.int32))  # Shape [3, 4], not [4, 3]

    server.shutdown()

  def test_stress_test(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([5], tf.int32)])
    def foo(x):
      return x + 1

    server.bind(foo, batched=True)
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

  @parameterized.parameters(([], False), ([1], True))
  def test_tpu(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    with tf.device('/device:CPU:0'):
      a = tf.Variable(1)

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      with tf.device('/device:CPU:0'):
        b = a + 1
        c = x + 1
      return x + b, c

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    a, b = client.foo(42)
    self.assertAllEqual(44, a)
    self.assertAllEqual(43, b)
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_tpu_tf_function_same_device(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    with tf.device('/device:CPU:0'):
      a = tf.Variable(1)

    with tf.device('/device:CPU:0'):

      @tf.function
      def get_a_plus_one():
        return a + 1

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      with tf.device('/device:CPU:0'):
        b = x + get_a_plus_one()
      return b + 1

    server.bind(foo, batched=batched)
    server.start()

    client = ops.Client(address)
    a = client.foo(1)
    self.assertAllEqual(4, a)
    server.shutdown()

  @parameterized.parameters(([], False), ([1], True))
  def test_bind_multiple_functions(self, dim, batched):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec(dim, tf.int32)])
    def foo(x):
      return x + 1

    @tf.function(input_signature=[
        tf.TensorSpec(dim, tf.int32),
        tf.TensorSpec(dim, tf.int32)
    ])
    def bar(x, y):
      return x * y

    server.bind(foo, batched=batched)
    server.bind(bar, batched=batched)
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


if __name__ == '__main__':
  tf.test.main()
