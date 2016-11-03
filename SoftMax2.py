import tensorflow as tf
import numpy as np
import Utility as util
import math


class SoftMax(object):
    def __init__(self, d, k):
        """
        :param d: dimensionality
        :param k: number of classes
        """

        self.D = d
        self.K = k
        self.NR_VALIDATION_DATA = 50  # the number of validation data
        self.NR_ITERATION = 1000  # the number of iteration in the SoftMax
        self.SHOW_ACC = 100  # Show Accuracy
        self.BATCH_SIZE = 5000  # the size of the batch
        self.TRAIN_STEP = 1e-2  # Train Step

    def training(self, features, labels):
        """
        Training Neural Network
        :param features: training data [50000 x 3072]
        :param labels: labels for X  [50000 x 1]
        :return: return a dictionary which contains all learned parameter
        """

        # Preprocessing
        features = self.__preprocessing(features)

        # Split data into training and validation sets.
        train_features = features[self.NR_VALIDATION_DATA:]
        train_labels = labels[self.NR_VALIDATION_DATA:]
        validation_features = features[0:self.NR_VALIDATION_DATA]
        validation_labels = labels[0:self.NR_VALIDATION_DATA]

        # Launch the session
        sess = tf.InteractiveSession()

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.D])  # the data
        y_ = tf.placeholder(tf.int64, shape=[None])  # the true labels

        # Initialize the variables
        W = tf.Variable(np.random.randn(self.D, self.K) * math.sqrt(2.0 / self.D * self.K), dtype=tf.float32)
        b = tf.Variable(tf.zeros([self.K]), dtype=tf.float32)

        # Calculate the output ( Soft Max )
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # Apply cross entropy loss ( loss data )
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # Training step - ADAM solver
        train_step = tf.train.AdamOptimizer(self.TRAIN_STEP).minimize(cross_entropy_mean)

        # Evaluate the model
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        for i in range(self.NR_ITERATION):
            batch = util.generate_batch(train_features, train_labels, self.BATCH_SIZE)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if i % self.SHOW_ACC == 0:
                value = accuracy.eval(feed_dict={x: validation_features, y_: validation_labels})
                print('Step - ', i, ' - Acc : ', value)

        W_final = W.eval()
        b_final = b.eval()

        # Close the session
        sess.close()

        return {'W': W_final, 'b': b_final}

    def predict(self, test_features, test_labels, nn):
        """
        Predict data
        :param test_features: testing data
        :param test_labels: labels for X
        :param nn: it is a dictionary which contains a Neural Network
        :return: return the predicted labels
        """

        # Preprocessing
        test_features = self.__preprocessing(test_features)

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.D])  # the data
        W = tf.placeholder(tf.float32, shape=[self.D, self.K])  # the weights
        b = tf.placeholder(tf.float32, shape=[self.K])  # the biases

        # Calculate the output ( Soft Max )
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # Launch the session
        sess = tf.InteractiveSession()

        # Run model on test data
        predicted_labels = sess.run(y, feed_dict={x: test_features, W: nn['W'], b: nn['b']})

        # Close the session
        sess.close()

        # Convert SoftMax predictions to label
        predicted_labels = np.argmax(predicted_labels, axis=1)

        # Calculate the accuracy
        acc = np.mean(predicted_labels == test_labels)

        print("The final accuracy is : ", acc)

        return predicted_labels

    def __preprocessing(self, X):
        """
        Preprocessing the X data by zero-centered and normalized them.
        :param X: the data.
        :return: return the new zero-centered and normalized data.
        """

        X = X.astype(np.float64)
        X = X.T - np.array(np.mean(X, axis=1, dtype=np.float64))  # zero-centered
        X = X / np.std(X.T, axis=1, dtype=np.float64)  # normalization
        X = X.T

        return X


if __name__ == "__main__":
    # Path to the saved learned parameter
    learn_data = 'result/SoftMax2/cifar_10'

    D = 3072  # dimensionality
    K = 10  # number of classes

    # Neural Network
    softMax = SoftMax(D, K)

    # Load the CIFAR10 data
    X, y, X_test, y_test = util.load_CIFAR10('data/')

    # Train the Neural Network
    if util.file_exist(learn_data):
        nn_parameter = util.unpickle(learn_data)
    else:
        nn_parameter = softMax.training(X, y)

        util.pickle_nn(learn_data, nn_parameter)

    # Test the Neural Network
    predicted_labels = softMax.predict(X_test, y_test, nn_parameter)

    # Save the predictions to label
    util.save_predicted_labels('result/SoftMax2/submission.csv', predicted_labels)
