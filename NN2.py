import tensorflow as tf
import numpy as np
import Utility as util
import math


class NeuralNetwork(object):
    def __init__(self, d, k):
        """
        :param d: dimensionality
        :param k: number of classes
        """

        self.D = d
        self.K = k
        self.H1 = 100  # size of first hidden layer
        self.H2 = 100  # size of second hidden layer
        self.NR_VALIDATION_DATA = 50  # the number of validation data
        self.NR_ITERATION = 100  # the number of iteration in the SoftMax
        self.SHOW_ACC = 10  # Show Accuracy
        self.BATCH_SIZE = 5000  # the size of the batch
        self.TRAIN_STEP = 1e-3  # Train Step

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

        # First layer
        W1 = tf.Variable(np.random.randn(self.D, self.H1) * math.sqrt(2.0 / self.D * self.H1), dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.H1]), dtype=tf.float32)

        # Second level
        W2 = tf.Variable(np.random.randn(self.H1, self.H2) * math.sqrt(2.0 / self.H1 * self.H2), dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.H2]), dtype=tf.float32)

        # Third level
        W3 = tf.Variable(np.random.randn(self.H2, self.K) * math.sqrt(2.0 / self.H2 * self.K), dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([self.K]), dtype=tf.float32)

        # Hidden Layer 1
        hidden1 = tf.add(tf.matmul(x, W1), b1)
        hidden1_relu = tf.nn.relu(hidden1)

        # Hidden Layer 2
        hidden2 = tf.add(tf.matmul(hidden1_relu, W2), b2)
        hidden2_relu = tf.nn.relu(hidden2)

        # Calculate the output ( Soft Max )
        y = tf.nn.softmax(tf.matmul(hidden2_relu, W3) + b3)

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

        W1_final = W1.eval()
        b1_final = b1.eval()

        W2_final = W2.eval()
        b2_final = b2.eval()

        W3_final = W3.eval()
        b3_final = b3.eval()

        # Close the session
        sess.close()

        return {'W1': W1_final, 'b1': b1_final, 'W2': W2_final, 'b2': b2_final, 'W3': W3_final, 'b3': b3_final}

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
        W1 = tf.placeholder(tf.float32, shape=[self.D, self.H1])  # the weights
        b1 = tf.placeholder(tf.float32, shape=[self.H1])  # the biases
        W2 = tf.placeholder(tf.float32, shape=[self.H1, self.H2])  # the weights
        b2 = tf.placeholder(tf.float32, shape=[self.H2])  # the biases
        W3 = tf.placeholder(tf.float32, shape=[self.H2, self.K])  # the weights
        b3 = tf.placeholder(tf.float32, shape=[self.K])  # the biases

        # Hidden Layer 1
        hidden1 = tf.add(tf.matmul(x, W1), b1)
        hidden1_relu = tf.nn.relu(hidden1)

        # Hidden Layer 2
        hidden2 = tf.add(tf.matmul(hidden1_relu, W2), b2)
        hidden2_relu = tf.nn.relu(hidden2)

        # Calculate the output ( Soft Max )
        y = tf.nn.softmax(tf.matmul(hidden2_relu, W3) + b3)

        # Launch the session
        sess = tf.InteractiveSession()

        # Initialize the placeholder
        feed_dict = {
            x: test_features,
            W1: nn['W1'],
            b1: nn['b1'],
            W2: nn['W2'],
            b2: nn['b2'],
            W3: nn['W3'],
            b3: nn['b3']
        }

        # Run model on test data
        predicted_labels = sess.run(y, feed_dict=feed_dict)

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
    learn_data = 'result/NN2/cifar_10'

    D = 3072  # dimensionality
    K = 10  # number of classes

    # Neural Network
    softMax = NeuralNetwork(D, K)

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
    util.save_predicted_labels('result/NN2/submission.csv', predicted_labels)
