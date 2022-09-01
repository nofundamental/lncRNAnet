import tensorflow as tf

print(tf.test.is_built_with_cuda())

# print(tf.config.list_physical_devices('GPU'))
print(tf.config.experimental.list_physical_devices('GPU'))

print(tf.__version__)