import tensorflow as tf
import os
import numpy as np
import cv2
from Generator import generator
from Discriminator import discriminator
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


class InfoGan:
    def __init__(self, sess, args):

        #########################
        #                       #
        #    General Setting    #
        #                       #
        #########################
        self.sess = sess
        self.args = args
        self.summary_dir = os.path.join(self.args.model_dir, 'log')
        self.test_dir = os.path.join(self.args.model_dir, 'test')

        if not self.args.model_dir:
            raise ValueError('Need to provide model directory')

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        #########################
        #                       #
        #     Model Building    #
        #                       #
        #########################

        # 1. Build Generator

        # Create latent variable
        with tf.name_scope('noise_sample'):
            self.z_cat = tf.placeholder(tf.int32, [None])
            self.z_cont = tf.placeholder(tf.float32, [None, args.num_cont])
            self.z_rand = tf.placeholder(tf.float32, [None, args.num_rand])

            z = tf.concat([tf.one_hot(self.z_cat, args.num_category), self.z_cont, self.z_rand], axis=1)
            self.z = tf.identity(z)

        self.g = generator(z, args)

        # 2. Build Discriminator

        # Real Data
        with tf.name_scope('data_and_target'):
            self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])

            y_real = tf.ones([tf.shape(self.x)[0]])
            y_fake = tf.zeros([tf.shape(self.x)[0]])

        d_real, _, _, _ = discriminator(self.x, args)
        d_fake, r_cat, r_cont_mu, r_cont_var = discriminator(self.g, args)

        # 3. Calculate loss

        # -log(D(G(x))) trick
        with tf.name_scope('loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=y_real))

            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=y_fake))
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=y_real))

            self.d_loss = (self.d_loss_fake + self.d_loss_real)
            
            # discrete logQ(c|x)
            self.cat_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=r_cat,
                                                                                          labels=self.z_cat))

            eplison = (r_cont_mu-self.z_cont) / r_cont_var

            # variance = 1
            # log gaussian distribution (continuous logQ(c|x))
            self.cont_loss = -tf.reduce_mean(tf.reduce_sum(-0.5*tf.log(2*np.pi*r_cont_var+1e-8)-0.5*tf.square(eplison), axis=1))

            self.train_g_loss = self.g_loss + self.cat_loss + self.cont_loss*0.1
            self.train_d_loss = self.d_loss + self.cat_loss + self.cont_loss*0.1

        # 4. Update weights
        g_param = tf.trainable_variables(scope='generator')
        d_param = tf.trainable_variables(scope='discriminator')

        with tf.name_scope('optimizer'):
            g_optim = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=0.5, beta2=0.99)
            self.g_train_op = g_optim.minimize(self.train_g_loss, var_list=g_param)
            d_optim = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=0.5, beta2=0.99)
            self.d_train_op = d_optim.minimize(self.train_d_loss, var_list=d_param)

        # 5. Visualize

        tf.summary.image('Real', self.x)
        tf.summary.image('Fake', self.g)

        with tf.name_scope('Generator'):
            tf.summary.scalar('g_total_loss', self.train_g_loss)
        with tf.name_scope('Discriminator'):
            tf.summary.scalar('d_total_loss', self.train_d_loss)
        with tf.name_scope('All_Loss'):
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)
            tf.summary.scalar('cat_loss', self.cat_loss)
            tf.summary.scalar('cont_loss', self.cont_loss)

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self):

        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(self.args.model_dir)
        if checkpoint:
            print('Load checkpoint {}...'.format(checkpoint))
            self.saver.restore(self.sess, checkpoint)

        steps_per_epoch = mnist.train.labels.shape[0] // self.args.batch_size
        counter = 0

        for epoch in range(self.args.epoch):
            for step in range(steps_per_epoch):
                x_batch, _ = mnist.train.next_batch(self.args.batch_size)
                x_batch = np.expand_dims(np.reshape(x_batch, [-1, 28, 28]), axis=-1)
                z_cont = np.random.uniform(-1, 1, size=[self.args.batch_size, self.args.num_cont])
                z_rand = np.random.uniform(-1, 1, size=[self.args.batch_size, self.args.num_rand])
                z_cat = np.random.randint(self.args.num_category, size=[self.args.batch_size])

                d_loss, _ = self.sess.run([self.train_d_loss, self.d_train_op], feed_dict={self.x: x_batch,
                                                                                           self.z_cat: z_cat,
                                                                                           self.z_cont: z_cont,
                                                                                           self.z_rand: z_rand})

                g_loss, _ = self.sess.run([self.train_g_loss, self.g_train_op],
                                             feed_dict={self.x: x_batch,
                                                        self.z_cat: z_cat,
                                                        self.z_cont: z_cont,
                                                        self.z_rand: z_rand})
                summary = self.sess.run(self.summary_op, feed_dict={self.x: x_batch,
                                                                    self.z_cat: z_cat,
                                                                    self.z_cont: z_cont,
                                                                    self.z_rand: z_rand})
                if step % 100 == 0:
                    print('Epoch[{}/{}] Step[{}/{}] g_loss:{:.4f}, d_loss:{:.4f}'.format(epoch, self.args.epoch, step,
                                                                                         steps_per_epoch, g_loss,
                                                                                         d_loss))

                summary_writer.add_summary(summary, counter)
                counter += 1

            self.save(counter)

    def inference(self):

        if self.args.model_dir is None:
            raise ValueError('Need to provide model directory')

        checkpoint = tf.train.latest_checkpoint(self.args.model_dir)

        if not checkpoint:
            raise FileNotFoundError('Checkpoint is not found in {}'.format(self.args.model_dir))
        else:
            print('Loading model checkpoint {}...'.format(self.args.model_dir))
            self.saver.restore(self.sess, checkpoint)

        for q in range(2):
            col = []
            for c in range(10):
                row = []
                for d in range(11):
                    z_cat = [c]
                    z_cont = -np.ones([1, self.args.num_cont])*2 + d*0.4
                    z_cont[0, q] = 0
                    z_rand = np.random.uniform(-1, 1, size=[1, self.args.num_rand])

                    g, z = self.sess.run([self.g, self.z], feed_dict={self.z_cat: z_cat,
                                                                      self.z_cont: z_cont,
                                                                      self.z_rand: z_rand})
                    g = np.squeeze(g)
                    multiplier = 255.0 / g.max()
                    g = (g * multiplier).astype(np.uint8)
                    row.append(g)

                row = np.concatenate(row, axis=1)
                col.append(row)
            result = np.concatenate(col, axis=0)
            filename = 'continuous_' + str(q) + '_col_cat_row_change.png'
            cv2.imwrite(os.path.join(self.test_dir, filename), result)

    def save(self, step):
        model_name = 'infogan.model'

        self.saver.save(self.sess, os.path.join(self.args.model_dir, model_name), global_step=step)
