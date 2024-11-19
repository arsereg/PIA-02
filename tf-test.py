import tensorflow as tf
import time

# Create a simple matrix operation
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    start = time.time()
    c = tf.matmul(a, b)
    print(f"Time taken on GPU: {time.time() - start} seconds")

with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    print(f"Time taken on CPU: {time.time() - start} seconds")

