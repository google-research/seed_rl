# TensorFlow gRPC

This directory contains a simple framework for creating servers and clients
with deep TensorFlow integration. It makes it possible to register a
`@tf.function` on a server which is called without interacting with Python
and thus avoids the global interpreter lock. Asynchronous streaming gRPC is used
which means the implementation can achieve up to a million QPS. Additionally, it
supports unix domain sockets that can be used for multi processing on a single
machine.

## Example

### Server

```python
server = grpc.Server([‘unix:/tmp/foo’, ‘localhost:8000’])

# This function is batched meaning it will be called once there are, in this
# case, 5 incoming calls.
@tf.function(input_signature=[tf.TensorSpec([5], tf.int32)])
def foo(x):
  return x + 1

server.bind(foo)

@tf.function(input_signature=[tf.TensorSpec([], tf.int32),
                              tf.TensorSpec([], tf.int32)])
def bar(x, y):
  return x + y

server.bind(bar)

server.start()
```

### Client

```python
client = grpc.Client(‘unix:/tmp/foo’)

# The following calls are TensorFlow operations which means they can be used
# eagerly with tensors or numpy arrays, but they can also be used in a
# `tf.function`.
client.foo(42)    # Returns 43
client.bar(1, 2)  # Returns 3
```
