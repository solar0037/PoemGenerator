import tensorflow as tf


def get_dataset(inputs, targets):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(64).shuffle(1000)
    return dataset
