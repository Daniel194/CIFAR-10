"""
ACCURACY : 68.5%
"""

import tensorflow as tf
import math
import time
import functools
import Utility
import numpy as np
import sys


class DigitsRecognition(object):
    def __init__(self):

        self.learning_rate = 1e-4  # Initial learning rate.
        self.epsilon = 1e-3  # Hyperparamter for Batch Normalization.

        self.max_steps = 20000  # Number of steps to run trainer.
        self.batch_size = 100  # Batch size.  Must divide evenly into the dataset sizes.

        self.IMAGE_SIZE = 32  # The size of the image in weight and height.
        self.NR_CHANEL = 3  # The number of chanel.
        self.IMAGE_SHAPE = (self.batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NR_CHANEL)

        self.W_conv1_shape = [5, 5, 3, 64]
        self.b_conv1_shape = [64]

        self.W_conv2_shape = [5, 5, 64, 128]
        self.b_conv2_shape = [128]

        self.W_conv3_shape = [5, 5, 128, 256]
        self.b_conv3_shape = [256]

        self.W_conv4_shape = [3, 3, 256, 256]
        self.b_conv4_shape = [256]

        self.W_conv5_shape = [3, 3, 256, 512]
        self.b_conv5_shape = [512]

        self.W_conv6_shape = [3, 3, 512, 512]
        self.b_conv6_shape = [512]

        self.W_fc1_shape = [8192, 2048]
        self.b_fc1_shape = [2048]

        self.W_fc2_shape = [2048, 10]
        self.b_fc2_shape = [10]

        self.dropout1 = 0.8
        self.dropout2 = 0.8
        self.dropout3 = 0.5

    def prediction(self, training, training_labels, validation, validation_labels, test):
        """
        Train CIFAR-10 for a number of steps.
        :param training: The training future.
        :param training_labels: The true labels for the training future.
        :param validation: The validation data.
        :param validation_labels: The true labels for validation labels.
        :param test: The test data.
        :return: Return all labels of test data.
        """

        # Preprocessing the training, validation adn test data.
        training = self.__data_preprocessing(training)
        validation = self.__data_preprocessing(validation)
        test = self.__data_preprocessing(test)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the images, labels and dropout probability.
            images_placeholder = tf.placeholder(tf.float32, shape=self.IMAGE_SHAPE)
            labels_placeholder = tf.placeholder(tf.int32, shape=self.batch_size)
            keep_prob1 = tf.placeholder(tf.float32)
            keep_prob2 = tf.placeholder(tf.float32)
            keep_prob3 = tf.placeholder(tf.float32)

            # Build a Graph that computes predictions from the inference model.
            logits = self.__inference(images_placeholder, keep_prob1, keep_prob2, keep_prob3)

            # Add to the Graph the Ops for loss calculation.
            loss = self.__loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.__training(loss)

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = self.__evaluation(logits, labels_placeholder)

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(tf.initialize_all_variables())

            # Start the training loop.
            for step in range(self.max_steps):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                images, images_labels = self.__generate_batch(training, training_labels)
                feed_dict = {images_placeholder: images,
                             labels_placeholder: images_labels,
                             keep_prob1: self.dropout1,
                             keep_prob2: self.dropout2,
                             keep_prob3: self.dropout3}

                # Run one step of the model.  The return values are the activations from the
                # `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                    sys.stdout.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == self.max_steps:
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')

                    self.__do_eval(sess, eval_correct, validation, validation_labels, images_placeholder,
                                   labels_placeholder, keep_prob1, keep_prob2, keep_prob3)

                    sys.stdout.flush()

            return self.__prediction(sess, logits, test, images_placeholder, keep_prob1, keep_prob2, keep_prob3)

    def __do_eval(self, sess, eval_correct, data, data_labels, images_placeholder, labels_placeholder, keep_prob1,
                  keep_prob2, keep_prob3):
        """
        Runs one evaluation against the full epoch of data.
        :param sess: The session in which the model has been trained.
        :param eval_correct: The Tensor that returns the number of correct predictions.
        :param data: The validation data.
        :param data_labels: The true label of validation data.
        :param images_placeholder: The images placeholder.
        :param labels_placeholder: The labels placeholder.
        :param keep_prob1: The probability to keep a neurone active.
        :param keep_prob2: The probability to keep a neurone active.
        :param keep_prob3: The probability to keep a neurone active.
        """

        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        num_examples = data.shape[0]

        for step in range(0, num_examples, self.batch_size):
            validation_batch = data[step:step + self.batch_size, :]
            validation_batch_labels = data_labels[step:step + self.batch_size]

            feed_dict = {images_placeholder: validation_batch,
                         labels_placeholder: validation_batch_labels,
                         keep_prob1: 1.0,
                         keep_prob2: 1.0,
                         keep_prob3: 1.0}
            true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = true_count / num_examples

        print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

    def __inference(self, images, keep_prob1, keep_prob2, keep_prob3):
        """
        Build the CIFAR-10 model up to where it may be used for inference.
        :param images: Images placeholder, from inputs().
        :param keep_prob1: the probability to keep a neuron data in Dropout Layer.
        :param keep_prob2: the probability to keep a neuron data in Dropout Layer.
        :param keep_prob3: the probability to keep a neuron data in Dropout Layer.
        :return: Output tensor with the computed logits.
        """

        # First Convolutional Layer
        with tf.name_scope('hidden1'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv1_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv1_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv1_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv1_shape), name='beta')

            z = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden1 = tf.nn.relu(bn)

        # Second Convolutional Layer
        with tf.name_scope('hidden2'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv2_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv2_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv2_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv2_shape), name='beta')

            z = tf.nn.conv2d(hidden1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden2 = tf.nn.relu(bn)

        # First Pool Layer
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Third Convolutional Layer
        with tf.name_scope('hidden3'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv3_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv3_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv3_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv3_shape), name='beta')

            z = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden3 = tf.nn.relu(bn)

        # Fourth Convolutional Layer
        with tf.name_scope('hidden4'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv4_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv4_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv4_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv4_shape), name='beta')

            z = tf.nn.conv2d(hidden3, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden4 = tf.nn.relu(bn)

        # Second Pool Layer
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(hidden4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First Dropout
        with tf.name_scope('dropout1'):
            dropout1 = tf.nn.dropout(pool2, keep_prob1)

        # Fifth Convolutional Layer
        with tf.name_scope('hidden5'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv5_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv5_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv5_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv5_shape), name='beta')

            z = tf.nn.conv2d(dropout1, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden5 = tf.nn.relu(bn)

        # First Dropout
        with tf.name_scope('dropout2'):
            dropout2 = tf.nn.dropout(hidden5, keep_prob2)

        # Sixth Convolutional Layer
        with tf.name_scope('hidden6'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_conv6_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_conv6_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_conv6_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_conv6_shape), name='beta')

            z = tf.nn.conv2d(dropout2, weights, strides=[1, 1, 1, 1], padding='SAME')
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            hidden6 = tf.nn.relu(bn)

        # Second Pool Layer
        with tf.name_scope('pool3'):
            pool3 = tf.nn.max_pool(hidden6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First Dropout
        with tf.name_scope('dropout3'):
            dropout3 = tf.nn.dropout(pool3, keep_prob3)

        # First Fully Connected Layer
        with tf.name_scope('fc1'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_fc1_shape)
            dropout3_flat = tf.reshape(dropout3, [-1, self.W_fc1_shape[0]])

            weights = tf.Variable(tf.truncated_normal(self.W_fc1_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            scale = tf.Variable(tf.ones(self.b_fc1_shape), name='scale')
            beta = tf.Variable(tf.zeros(self.b_fc1_shape), name='beta')

            z = tf.matmul(dropout3_flat, weights)
            batch_mean, batch_var = tf.nn.moments(z, [0])
            bn = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, self.epsilon)
            fc1 = tf.nn.relu(bn)

        # Second Fully Connected Layer
        with tf.name_scope('fc2'):
            nr_units = functools.reduce(lambda x, y: x * y, self.W_fc2_shape)

            weights = tf.Variable(tf.truncated_normal(self.W_fc2_shape, stddev=1.0 / math.sqrt(float(nr_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros(self.b_fc2_shape), name='biases')
            fc2 = tf.matmul(fc1, weights) + biases

        return fc2

    @staticmethod
    def __loss(softmax_logits, true_labels):
        """
        Calculates the loss from the logits and the labels.
        :param softmax_logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param true_labels: Labels tensor, int32 - [batch_size].
        :return: Loss tensor of type float.
        """

        true_labels = tf.to_int64(true_labels)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(softmax_logits, true_labels, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        return loss

    def __training(self, loss):
        """
        Sets up the training Ops.
        :param loss: Loss tensor, from loss().
        :return: The Op for training.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    @staticmethod
    def __evaluation(logits, true_labels):
        """
        Evaluate the quality of the logits at predicting the label.
        :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        :param true_labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
        :return: A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
        """

        # Top k correct prediction
        correct = tf.nn.in_top_k(logits, true_labels, 1)

        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def __generate_batch(self, data, data_labels):
        """
        Generate Batches.
        :param data: the features data.
        :param data_labels: the labels data.
        :return: return labels and features.
        """

        batch_indexes = np.random.random_integers(0, len(data) - 1, self.batch_size)
        batch_dat = data[batch_indexes]
        batch_labels = data_labels[batch_indexes]

        return batch_dat, batch_labels

    def __data_preprocessing(self, data):
        """
        Preprocesing the CIFAR-10 data.
        :param data: the data.
        :return: the zero-centered and normalization data.
        """

        data = data.astype(np.float64)
        data -= np.mean(data, dtype=np.float64)  # zero-centered
        data /= np.std(data, dtype=np.float64)  # normalization

        return np.reshape(data, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NR_CHANEL))

    def __prediction(self, sess, logits, data, images_placeholder, keep_prob1, keep_prob2, keep_prob3):
        """
        Predicting the labels of the data.
        :param sess: The session in which the model has been trained.
        :param logits: The tenssor that calculate the logits.
        :param data: The data.
        :param images_placeholder: The images placeholder.
        :param keep_prob1: The probability to keep a neurone active.
        :param keep_prob2: The probability to keep a neurone active.
        :param keep_prob3: The probability to keep a neurone active.
        :return: return the labels predicted.
        """

        steps_per_epoch = data.shape[0] // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        predicted_labels = []

        for step in range(0, num_examples, self.batch_size):
            data_batch = data[step:step + self.batch_size, :]
            feed_dict = {images_placeholder: data_batch,
                         keep_prob1: 1.0,
                         keep_prob2: 1.0,
                         keep_prob3: 1.0}

            # Run model on test data
            softmax = tf.nn.softmax(logits)
            batch_predicted_labels = sess.run(softmax, feed_dict=feed_dict)

            # Convert softmax predictions to label and append to all results.
            batch_predicted_labels = np.argmax(batch_predicted_labels, axis=1)
            predicted_labels.extend(batch_predicted_labels)

        sess.close()

        return predicted_labels


if __name__ == '__main__':
    # CONSTANTS
    SAVE_DATA = 'result/CNN3/submission.csv'
    OUTPUT_FILE = 'result/CNN3/output_cnn3.txt'
    DATA = 'data/'

    # Redirect the output to a file
    sys.stdout = open(OUTPUT_FILE, 'w')

    # Read the feature and the labels.
    features, labels, test_features, test_labels = Utility.load_CIFAR10(DATA)

    # Separate the training and validation data.
    NR_VAL = int(features.shape[0] * 0.1)

    train_features = features[NR_VAL:]
    train_labels = labels[NR_VAL:]
    validation_features = features[0:NR_VAL]
    validation_features_labels = labels[0:NR_VAL]

    model = DigitsRecognition()

    predictions = model.prediction(train_features, train_labels, validation_features,
                                   validation_features_labels, test_features)

    # Save the predictions to label
    Utility.create_file(SAVE_DATA)

    Utility.write_to_file(SAVE_DATA, predictions)

    # Calculate the accuracy
    test_labels = np.array(test_labels)
    acc = np.mean(predictions == test_labels)

    print('===================================')
    print("The final accuracy is : ", acc)
    print('===================================')

    print('DONE !')
