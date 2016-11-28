import train_image_classifier as base
import tensorflow as tf

def identity(input, redundant):
    return tf.identity(input)

class s:
    @staticmethod
    def one_hot_encoding(input, redundant):
        return tf.identity(input)


def main(_):
    print('starting!!')
    base.slim.one_hot_encoding = identity
    base.main(_)

if __name__ == '__main__':
    tf.app.run()
