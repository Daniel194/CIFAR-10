import numpy as np
import math
import Utility as util


class SoftMax(object):
    def __init__(self, d, k):
        """
        :param d: dimensionality
        :param k: number of classes
        """

        self.D = d
        self.K = k
        self.REG = 1e-3  # regularization strength
        self.DELTA = 1.0  # used in SVM Loss function.
        self.MU = 0.9  # parameter for parameter update ( SGD + Nesterov Momentum )
        self.LEARNING_RATE = 1e-2  # learning rate
        self.NR_ITERATION = 100  # the number of iteration in the Neural Network
        self.DISPLAY_LOSS = 10  # when to display loss score

    def training(self, X, y):
        """
        Training Neural Network
        :param X: training data [50000 x 3072]
        :param y: labels for X  [50000 x 1]
        :return: return a dictionary which contains all learned parameter
        """

        # Initialize parameters
        W1 = np.random.randn(self.D, self.K) * math.sqrt(2.0 / (self.D * self.K))  # [3072 x 10]
        b1 = np.zeros((1, self.K))  # [1 x 10]
        v1 = np.zeros(W1.shape)

        # Preprocessing the data
        X = self.__preprocessing(X)

        for i in range(self.NR_ITERATION):
            # Output neuron
            scores = np.dot(X, W1) + b1  # [50000 x 10]

            # Compute the loss
            loss_data, probs = self.__loss_data(scores, y)
            loss = loss_data + self.__regularization(W1)

            if i % self.DISPLAY_LOSS == 0:
                print("Iteration - ", i, " - Loss : ", loss)

            # Compute gradients
            dscores = self.__gradient(y, probs)  # [50000 x 10]

            # Backward pass - second layer
            dW1 = np.dot(X.T, dscores)  # [3072 x 10]
            db1 = np.sum(dscores, axis=0, keepdims=True)  # [1 x 10]

            # Perform parameter update ( SGD + Nesterov Momentum )
            v1_prev = v1  # back this up
            v1 = self.MU * v1 - self.LEARNING_RATE * dW1  # velocity update
            W1 += -self.MU * v1_prev + (1 + self.MU) * v1  # position update
            b1 += -self.LEARNING_RATE * db1

        return {'W1': W1, 'b1': b1}

    def predict(self, X, y, nn):
        """
        Predict data
        :param X: testing data
        :param y: labels for X
        :param nn: it is a dictionary which contains a Neural Network
        :return: return the predicted labels
        """

        # Preprocessing the data
        X = self.__preprocessing(X)

        # The Neural Network parameter
        W1 = nn['W1']
        b1 = nn['b1']

        # Output neuron
        out = np.dot(X, W1) + b1  # [50000 x 10]

        predicted_labels = np.argmax(out, axis=1)  # [50000 x 1]

        acc = np.mean(predicted_labels == y)

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

    def __regularization(self, W):
        """
        The L2 regularization function
        :param W: the weights
        :return: return a scalar data [1x1]
        """

        return 0.5 * self.REG * np.sum(W * W)

    def __loss_data(self, scores, y):
        """
        The Soft Max loss data function.
        :param scores: the scores obtained  by the neurla network [50000 x 10]
        :param y: the labels
        :return: the score of the Soft Max loss function and the probabilities
        """

        num_examples = y.shape[0]

        # get unnormalized probabilities
        exp_scores = np.exp(scores)

        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        corect_logprobs = -np.log(probs[range(num_examples), y])

        loss_data = np.sum(corect_logprobs) / num_examples

        return loss_data, probs

    def __gradient(self, y, probs):
        """
        Calculate the gradient of the Soft Max loss function.
        :param y: the true labels [50000 x 1]
        :param probs: scores obtain by the Neural Network [50000 x 10]
        :return: return the gradient of the scores [ 50000 x 10]
        """

        num_examples = y.shape[0]

        dscores = probs
        dscores[range(num_examples), y] -= 1

        dscores /= num_examples

        return dscores


if __name__ == '__main__':
    # path to the saved learned parameter
    learn_data = 'result/SoftMax1/cifar_10'

    D = 3072  # dimensionality
    K = 10  # number of classes

    # Neural Network
    nn = SoftMax(D, K)

    # load the CIFAR10 data
    X, y, X_test, y_test = util.load_CIFAR10('data/')

    # Train the Neural Network
    if util.file_exist(learn_data):
        nn_parameter = util.unpickle(learn_data)
    else:
        nn_parameter = nn.training(X, y)

        util.pickle_nn(learn_data, nn_parameter)

    # Test the Neural Network
    predicted_labels = nn.predict(X_test, y_test, nn_parameter)

    # Save the predictions to label
    util.save_predicted_labels('result/SoftMax1/submission.csv', predicted_labels)
