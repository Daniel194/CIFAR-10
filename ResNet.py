"""
AUTHOR : Lungu Daniel

ACCURACY : ??.? %
"""

import cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd


class ImageRecognition(object):
    def __init__(self, image_size=24, num_classes=10, batch_size=128, channels=3, layers=18):
        self._image_size = image_size
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._channels = channels
        self._layers = layers

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._images = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._global_step = tf.Variable(0, tf.int64, name="global_step")
        self._logits = self.__inference(self._images, self._keep_prob)
        self._avg_loss = self.__loss(self._labels, self._logits)
        self._train_op = self.__train(self._avg_loss)
        self._accuracy = self.__accuracy_score(self._labels, self._logits)
        self._saver = tf.train.Saver(tf.global_variables())
        self._session.run(tf.global_variables_initializer())

    def load_cifar10(self, is_train=True):

        if is_train:
            filenames = ["data/data_batch_%d" % j for j in range(1, 6)]
        else:
            filenames = ["data/test_batch"]

        images, labels = [], []

        for filename in filenames:
            cifar10 = self.__unpickle(filename)

            for i in range(len(cifar10["labels"])):
                images.append(self.__distort(cifar10["data"][i], is_train))

            labels += cifar10["labels"]

        return self.__shuffle(images, np.asarray(labels))

    def __distort(self, image, is_train=True):
        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(float)

        if is_train:
            image = self.__random_crop(image, (24, 24, 3))
            image = self.__random_flip_left_right(image)

        else:
            image = self.__crop_to_bounding_box(image, 4, 4, 24, 24)

        image = self.__per_image_whitening(image)

        return image

    @staticmethod
    def __unpickle(filename):
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            dict = u.load()

            return dict

    @staticmethod
    def __shuffle(images, labels):
        perm = np.arange(len(labels))
        np.random.shuffle(perm)

        return np.asarray(images)[perm], np.asarray(labels)[perm]

    @staticmethod
    def __dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    @staticmethod
    def __crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
        return image[offset_width:offset_width + target_width, offset_height:offset_height + target_height]

    @staticmethod
    def __random_contrast(image, lower, upper, seed=None):
        contrast_factor = np.random.uniform(lower, upper)
        avg = np.mean(image)

        return (image - avg) * contrast_factor + avg

    @staticmethod
    def __random_brightness(image, max_delta, seed=None):
        delta = np.random.randint(-max_delta, max_delta)

        return image - delta

    @staticmethod
    def __random_blur(image, size):
        if np.random.random() < 0.5:
            image = cv2.blur(image, size)

        return image

    @staticmethod
    def __normalize(image):
        return image / 255.0

    @staticmethod
    def __per_image_whitening(image):
        return (image - np.mean(image)) / np.std(image)

    @staticmethod
    def __random_flip_left_right(image):
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)

        return image

    @staticmethod
    def __random_flip_up_down(image):
        if np.random.random() < 0.5:
            image = cv2.flip(image, 0)

        return image

    @staticmethod
    def __random_crop(image, size):
        if len(image.shape):
            W, H, D = image.shape
            w, h, d = size
        else:
            W, H = image.shape
            w, h = size

        left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)

        return image[left:left + w, top:top + h]

    @staticmethod
    def __conv2d(x, W, strides=1):
        return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    @staticmethod
    def __weight_variable(shape, wd=1e-4):
        k, c = 3, shape[-2]
        var = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / (k * k * c))))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
            tf.add_to_collection('losses', weight_decay)

        print(var.get_shape())

        return var

    @staticmethod
    def __bias_variable(shape):
        b = tf.Variable(tf.constant(0.0, shape=shape))
        print("bias", b.get_shape())

        return b

    def __conv(self, x, n, strides=1, bias_term=True):
        W = self.__weight_variable([3, 3, self.__channels(x), n])
        res = self.__conv2d(x, W, strides)

        if bias_term:
            res += self.__bias_variable([n])

        return res

    def __dense(self, x, n):
        W, b = self.__weight_variable([self.__volume(x), n]), self.__bias_variable([n])

        return tf.matmul(x, W) + b

    @staticmethod
    def __activation(x):
        print("ReLU")

        return tf.nn.relu(x)

    @staticmethod
    def __max_pool(x, ksize=2, strides=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME')

    @staticmethod
    def __avg_pool(x, ksize=2, strides=2):
        return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME')

    @staticmethod
    def __channels(x):
        return int(x.get_shape()[-1])

    @staticmethod
    def __volume(x):
        return np.prod([d for d in x.get_shape()[1:].as_list()])

    def __flatten(self, x):
        return tf.reshape(x, [-1, self.__volume(x)])

    def __batch_normalization(self, x):
        print("Batch Norm")

        eps = 1e-5
        gamma = tf.Variable(tf.constant(1.0, shape=[self.__channels(x)]))
        beta = tf.Variable(tf.constant(0.0, shape=[self.__channels(x)]))
        mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)

        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)

    @staticmethod
    def __accuracy_score(labels, logits):
        correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return accuracy

    def fit(self, X, y, max_epoch=10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                batch_images, batch_labels = X[i:i + self._batch_size], y[i:i + self._batch_size]
                feed_dict = {self._images: batch_images, self._labels: batch_labels, self._keep_prob: 0.5}
                _, train_avg_loss, global_step = self._session.run(
                    fetches=[self._train_op, self._avg_loss, self._global_step], feed_dict=feed_dict)

            print("epochs =", global_step)

    def __predict(self, X):
        return np.argmax(self.__predict_proba(X), axis=1)

    def __predict_proba(self, X):
        res = None

        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i + self._batch_size]
            feed_dict = {self._images: batch_images, self._keep_prob: 1.0}
            test_logits = self._session.run(fetches=self._logits, feed_dict=feed_dict)
            if res is None:
                res = test_logits
            else:
                res = np.r_[res, test_logits]

        return res

    def score(self, X, y):
        total_acc, total_loss = 0, 0

        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i + self._batch_size], y[i:i + self._batch_size]
            feed_dict = {self._images: batch_images, self._labels: batch_labels, self._keep_prob: 1.0}
            acc, avg_loss = self._session.run(fetches=[self._accuracy, self._avg_loss], feed_dict=feed_dict)
            total_acc += acc * len(batch_images)
            total_loss += avg_loss * len(batch_images)

        return total_acc / len(X), total_loss / len(X)

    def save(self, filepath):
        self._saver.save(self._session, filepath)

    def __loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)

        return tf.add_n(tf.get_collection('losses'))

    def __residual(self, h, channels, strides):
        h0 = h
        h1 = self.__activation(self.__batch_normalization(self.__conv(h0, channels, strides, bias_term=False)))
        h2 = self.__batch_normalization(self.__conv(h1, channels, bias_term=False))

        if self.__volume(h0) == self.__volume(h2):
            h = h2 + h0
        else:
            h3 = self.__avg_pool(h0)
            h4 = tf.pad(h3, [[0, 0], [0, 0], [0, 0], [int(channels / 4), int(channels / 4)]])
            h = h2 + h4

        return self.__activation(h)

    def __inference(self, X, keep_prob):
        h = X
        h = self.__activation(self.__batch_normalization(self.__conv(h, 16, bias_term=False)))

        for i in range(self._layers):
            h = self.__residual(h, channels=16, strides=1)

        for channels in [32, 64]:
            for i in range(self._layers):
                strides = 2 if i == 0 else 1
                h = self.__residual(h, channels, strides)

        h = tf.reduce_mean(h, reduction_indices=[1, 2])
        h = self.__dense(h, self._num_classes)

        return h

    def __train(self, avg_loss):
        lr = tf.where(tf.less(self._global_step, 32000), 0.1,
                      tf.where(tf.less(self._global_step, 48000), 0.01, 0.001))

        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(avg_loss,
                                                                                   global_step=self._global_step)


def run():
    model = ImageRecognition()

    test_images, test_labels = model.load_cifar10(is_train=False)
    records = []

    for epoch in range(200):
        train_images, train_labels = model.load_cifar10(is_train=True)
        num_epoch = 1
        start_time = time.time()
        model.fit(train_images, train_labels, max_epoch=num_epoch)
        duration = time.time() - start_time
        examples_per_sec = (num_epoch * len(train_images)) / duration
        train_accuracy, train_loss = model.score(train_images, train_labels)
        test_accuracy, test_loss = model.score(test_images, test_labels)

        summary = {
            "epoch": epoch,
            "name": model.__class__.__name__,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "examples_per_sec": examples_per_sec,
        }

        print("[%(epoch)d][%(name)s]train-acc: %(train_accuracy).3f, train-loss: %(train_loss).3f," % summary)
        print(
            "test-acc: %(test_accuracy).3f, test-loss: %(test_loss).3f, %(examples_per_sec).1f examples/sec" % summary)

        records.append(summary)
        df = pd.DataFrame(records)
        df.to_csv("result/%s.csv" % model.__class__.__name__.lower(), index=False)

        if df["test_accuracy"].max() - 1e-5 < test_accuracy:
            save_dir = "result/%s" % model.__class__.__name__
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            print("Save to %s" % save_dir)
            model.save(save_dir + "/model.ckpt")

        if train_loss * 300 < test_loss:  # Overfitting
            break


if __name__ == "__main__":
    run()
