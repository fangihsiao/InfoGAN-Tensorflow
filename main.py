import tensorflow as tf
from model import InfoGan

flags = tf.app.flags

flags.DEFINE_string('model_dir', None, 'model directory')
flags.DEFINE_boolean('is_training', None, 'whether it is training or inferecing')
flags.DEFINE_boolean('fix_var', True, 'whether to approximate variance')
flags.DEFINE_integer('num_category', 10, 'category dim of latent variable')
flags.DEFINE_integer('num_cont', 2, 'continuous dim of latent variable')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epochs')
flags.DEFINE_integer('num_rand', 62, 'random noise dim of latent variable')
flags.DEFINE_float('d_lr', 2e-4, 'learning rate for discriminator')
flags.DEFINE_float('g_lr', 1e-3, 'learning rate for generator')

FLAGS = flags.FLAGS


def main(unused_argv):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = InfoGan(sess, FLAGS)

    model.train() if FLAGS.is_training else model.inference()


if __name__ == '__main__':
    tf.app.run()
