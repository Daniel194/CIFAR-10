import numpy as np
import tensorflow as tf
import Utility as util
import math


class ConvolutionNeuralNetwork:
    def __init__(self, d, k):
        """
         :param d: dimensionality
         :param k: number of classes
        """

        # Parameter
        self.D = d
        self.K = k
        self.NR_VALIDATION_DATA = 50
        self.NR_ITERATION = 200
        self.BATCH_SIZE = 50
        self.SHOW_ACC = 10
        self.TRAIN_STEP = 1e-4

        # Shape
        self.W1_SHAPE = [5, 5, 3, 16]
        self.B1_SHAPE = [16]
        self.W2_SHAPE = [5, 5, 16, 20]
        self.B2_SHAPE = [20]
        self.W3_SHAPE = [5, 5, 20, 20]
        self.B3_SHAPE = [20]
        self.WFC_SHAPE = [320, k]
        self.BFC_SHAPE = [k]

    def training(self, features, labels):
        """
        Training the Convolutional Neural Network
        :param features: the training data [50000 x 3072]
        :param labels: the true label for X  [50000 x 1]
        :return: return a dictionary which contains all learned parameters
        """

        # Preprocessing the data
        features = self.__preprocessing(features)  # [50000 x 3072]

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

        # Reshape
        x_image = tf.reshape(x, [-1, 32, 32, 3])

        # Initialize the weights and the biases
        # First Layer
        W1 = self.__weight_variable(self.W1_SHAPE)  # [ 5 x 5 x 3 x 16 ]
        b1 = self.__bias_variable(self.B1_SHAPE)  # [16]
        # Second Layer
        W2 = self.__weight_variable(self.W2_SHAPE)  # [ 5 x 5 x 16 x 20 ]
        b2 = self.__bias_variable(self.B2_SHAPE)  # [20]
        # Third Layer
        W3 = self.__weight_variable(self.W3_SHAPE)  # [ 5 x 5 x 20 x 20 ]
        b3 = self.__bias_variable(self.B3_SHAPE)  # [20]
        # Full Connected Layer
        WFC = self.__weight_variable(self.WFC_SHAPE)  # [ 160 x 10 ]
        bFC = self.__bias_variable(self.BFC_SHAPE)  # [10]

        # Calculate the hidden layers
        # First Layer
        H1_conv = self.__activation(self.__convolution(x_image, W1) + b1)  # [50000 x 32 x 32 x 16]
        H1_pool = self.__pool(H1_conv)  # [50000 x 16 x 16 x 16]
        # Second Layer
        H2_conv = self.__activation(self.__convolution(H1_pool, W2) + b2)  # [50000 x 16 x 16 x 20]
        H2_pool = self.__pool(H2_conv)  # [50000 x 8 x 8 x 20]
        # Third Layer
        H3_conv = self.__activation(self.__convolution(H2_pool, W3) + b3)  # [50000 x 8 x 8 x 20]
        H3_pool = self.__pool(H3_conv)  # [50000 x 4 x 4 x 20]
        # Full Connected Layer
        H3_pool_flatten = tf.reshape(H3_pool, [-1, self.WFC_SHAPE[0]])  # [ 50000 x 160 ]
        HFC = tf.matmul(H3_pool_flatten, WFC) + bFC  # [ 50000 x 10 ]

        # Calculate the output
        y_conv = tf.nn.softmax(HFC)  # [ 50000 x 10]

        # Apply cross entropy loss ( loss data )
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # Training step - ADAM solver
        train_step = tf.train.AdamOptimizer(self.TRAIN_STEP).minimize(cross_entropy_mean)

        # Evaluate the model
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        for i in range(self.NR_ITERATION):
            batch = util.generate_batch(train_features, train_labels, self.BATCH_SIZE)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if i % self.SHOW_ACC == 0:
                train_accuracy = accuracy.eval(feed_dict={x: validation_features, y_: validation_labels})

                print('Step - ', i, ' - Acc : ', train_accuracy)

        W1_final = W1.eval()
        b1_final = b1.eval()

        W2_final = W2.eval()
        b2_final = b2.eval()

        W3_final = W3.eval()
        b3_final = b3.eval()

        WFC_final = WFC.eval()
        bFC_final = bFC.eval()

        # Close the session
        sess.close()

        return {
            'W1': W1_final,
            'b1': b1_final,
            'W2': W2_final,
            'b2': b2_final,
            'W3': W3_final,
            'b3': b3_final,
            'WFC': WFC_final,
            'bFC': bFC_final
        }

    def predict(self, test_features, test_labels, nn):
        """
        Predict data
        :param test_features: testing data
        :param test_labels: labels for test_features
        :param nn: it is a dictionary which contains a Neural Network
        :return: return the predicted labels and the accuracy
        """

        # Preprocessing
        test_features = self.__preprocessing(test_features)

        # Placeholders
        x = tf.placeholder(tf.float32, shape=[None, self.D])  # the data
        W1 = tf.placeholder(tf.float32, shape=self.W1_SHAPE)  # the weights
        b1 = tf.placeholder(tf.float32, shape=self.B1_SHAPE)  # the biases
        W2 = tf.placeholder(tf.float32, shape=self.W2_SHAPE)  # the weights
        b2 = tf.placeholder(tf.float32, shape=self.B2_SHAPE)  # the biases
        W3 = tf.placeholder(tf.float32, shape=self.W3_SHAPE)  # the weights
        b3 = tf.placeholder(tf.float32, shape=self.B3_SHAPE)  # the biases
        WFC = tf.placeholder(tf.float32, shape=self.WFC_SHAPE)  # the weights
        bFC = tf.placeholder(tf.float32, shape=self.BFC_SHAPE)  # the biases

        # Reshape
        x_image = tf.reshape(x, [-1, 32, 32, 3])

        # Calculate the hidden layers
        # First Layer
        H1_conv = self.__activation(self.__convolution(x_image, W1) + b1)  # [50000 x 32 x 32 x 16]
        H1_pool = self.__pool(H1_conv)  # [50000 x 16 x 16 x 16]
        # Second Layer
        H2_conv = self.__activation(self.__convolution(H1_pool, W2) + b2)  # [50000 x 16 x 16 x 20]
        H2_pool = self.__pool(H2_conv)  # [50000 x 8 x 8 x 20]
        # Third Layer
        H3_conv = self.__activation(self.__convolution(H2_pool, W3) + b3)  # [50000 x 8 x 8 x 20]
        H3_pool = self.__pool(H3_conv)  # [50000 x 4 x 4 x 20]
        # Full Connected Layer
        H3_pool_flatten = tf.reshape(H3_pool, [-1, self.WFC_SHAPE[0]])  # [ 50000 x 160 ]
        HFC = tf.matmul(H3_pool_flatten, WFC) + bFC  # [ 50000 x 10 ]

        # Calculate the output
        y_conv = tf.nn.softmax(HFC)  # [ 50000 x 10]

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
            b3: nn['b3'],
            WFC: nn['WFC'],
            bFC: nn['bFC']
        }

        # Run model on test data
        predicted_labels = sess.run(y_conv, feed_dict=feed_dict)

        # Close the session
        sess.close()

        # Convert SoftMax predictions to label
        predicted_labels = np.argmax(predicted_labels, axis=1)

        # Calculate the accuracy
        acc = np.mean(predicted_labels == test_labels)

        return predicted_labels, acc

    def __preprocessing(self, X):
        """
        Preprocessing the X data by zero-centered and normalized them.
        :param X: the data.
        :return: return the new zero-centered and normalized data.
        """

        X = X.astype(np.float64)
        X = X - np.mean(X, dtype=np.float64)  # zero-centered
        X = X / np.std(X, dtype=np.float64)  # normalization

        return X

    def __weight_variable(self, shape):
        """
        Initialize the weights variable.
        :param shape: the shape.
        :return: return a TensorFlow variable
        """

        if len(shape) == 4:
            initial = np.random.randn(shape[0], shape[1], shape[2], shape[3]) * math.sqrt(
                2.0 / (shape[0] * shape[1] * shape[2] * shape[3]))
        else:
            initial = np.random.randn(shape[0], shape[1]) * math.sqrt(2.0 / (shape[0] * shape[1]))

        return tf.Variable(initial, dtype=tf.float32)

    def __bias_variable(self, shape):
        """
        Initialize the biases variable.
        :param shape:t he shape.
        :return: return a TensorFlow variable
        """

        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)

    def __convolution(self, x, W):
        """
        The convolution layer calculation.
        :param x: the data.
        :param W: the weights.
        :return: return the output of the convolution layer.
        """

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __activation(self, x):
        """
        The activation function.
        :param x: the data.
        :return: return the data after apply the activation.
        """

        return tf.nn.relu(x)

    def __pool(self, x):
        """
        The pool layer.
        :param x: the data.
        :return: return the output of the pool layer.
        """

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # Variable
    D = 3072  # dimensionality
    K = 10  # number of classes
    learn_data = 'result/CNN1/cifar_10'
    final_accuracy = 0
    batch_size = 50

    # Neural Network
    cnn = ConvolutionNeuralNetwork(3072, 10)

    # Load the CIFAR10 data
    X, y, X_test, y_test = util.load_CIFAR10('data/')
    # X_test = X_test[0:3 * batch_size]
    # y_test = y_test[0:3 * batch_size]

    # Train the Neural Network
    if util.file_exist(learn_data):
        nn_parameter = util.unpickle(learn_data)
    else:
        nn_parameter = cnn.training(X, y)

        util.pickle_nn(learn_data, nn_parameter)

    util.create_file('result/CNN1/submission.csv')

    for i in range(0, X_test.shape[0], batch_size):
        batch_test_feature = X_test[i:i + batch_size, :]
        batch_test_labels = y_test[i:i + batch_size]

        # Test the Neural Network
        predicted_labels, accuracy = cnn.predict(batch_test_feature, batch_test_labels, nn_parameter)

        final_accuracy += accuracy

        # Save the predictions to label
        util.append_data_to_file('result/CNN1/submission.csv', predicted_labels, i)

    nr_iteration = X_test.shape[0] / batch_size

    final_accuracy /= nr_iteration

    print("The final accuracy is : ", final_accuracy)
