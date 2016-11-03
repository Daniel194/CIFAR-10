import numpy as np
import math
import Utility as util


class NeuralNetwork(object):
    def __init__(self, d, k):
        """
        :param d: dimensionality
        :param k: number of classes
        """

        self.D = d
        self.K = k
        self.h1 = 100  # size of first hidden layer
        self.h2 = 100  # size of second hidden layer
        self.reg = 1e-3  # regularization strength
        self.p = 0.5  # probability of keeping a unit active.
        self.delta = 1.0  # used in SVM Loss function.
        self.mu = 0.9  # parameter for parameter update ( SGD + Nesterov Momentum )
        self.learning_rate = 1e-1  # learning rate
        self.number_of_iteration = 200  # the number of iteration in the Neural Network
        self.display_loss = 1  # when to display loss score

    def training(self, X, y):
        """
        Training Neural Network
        :param X: training data [50000 x 3072]
        :param y: labels for X  [50000 x 1]
        :return: return a dictionary which contains all learned parameter
        """

        # Initialize parameters

        # First layer
        W1 = np.random.randn(self.D, self.h1) * math.sqrt(2.0 / (self.D * self.h1))  # [3072 x 100]
        b1 = np.zeros((1, self.h1))  # [1 x 100]
        v1 = np.zeros(W1.shape)

        # Second layer
        W2 = np.random.randn(self.h1, self.h2) * math.sqrt(2.0 / (self.h1 * self.h2))  # [100 x 100]
        b2 = np.zeros((1, self.h2))  # [1 x 100]
        v2 = np.zeros(W2.shape)

        # Third layer
        W3 = np.random.randn(self.h2, self.K) * math.sqrt(2.0 / (self.h2 * self.K))  # [100 x 10]
        b3 = np.zeros((1, self.K))  # [1 x 10]
        v3 = np.zeros(W3.shape)

        # Preprocessing the data
        X = self.__preprocessing(X)

        for i in range(self.number_of_iteration):
            # calculate first hidden layer activations
            hidden_layer1 = self.__forward(X, W1, b1)  # [50000 x 100]
            # u1 = (np.random.rand(*hidden_layer1.shape) < self.p) / self.p  # first dropout mask.
            # hidden_layer1 *= u1  # drop!

            # calculate second hidden layer activations
            hidden_layer2 = self.__forward(hidden_layer1, W2, b2)  # [50000 x 100]
            # u2 = (np.random.rand(*hidden_layer2.shape) < self.p) / self.p  # second dropout mask.
            # hidden_layer2 *= u2  # drop!

            # output neuron
            scores = np.dot(hidden_layer2, W3) + b3  # [50000 x 10]

            # compute the loss
            loss_data, probs = self.__loss_data(scores, y)
            loss = loss_data + self.__regularization(W3)

            if i % self.display_loss == 0:
                print("Iteration - ", i, " - Loss : ", loss)

            # compute gradients
            dscores = self.__gradient(y, probs)  # [50000 x 10]

            # backward pass - third layer
            dW3 = np.dot(hidden_layer2.T, dscores)  # [100 x 10]
            dW3 += self.reg * W3
            db3 = np.sum(dscores, axis=0, keepdims=True)  # [1 x 10]

            # next backprop into second hidden layer
            dhidden2 = np.dot(dscores, W3.T)  # [50000 x 100]

            # backprop the ReLU non-linearity
            dhidden2[hidden_layer2 <= 0] = 0

            # backward pass - second layer
            dW2 = np.dot(hidden_layer1.T, dhidden2)  # [100 x 100]
            # dW2 += self.reg * W2
            db2 = np.sum(dhidden2, axis=0, keepdims=True)  # [1 x 100]

            # next backprop into first hidden layer
            dhidden1 = np.dot(dhidden2, W2.T)  # [50000 x 100]

            # backprop the ReLU non-linearity
            dhidden1[hidden_layer1 <= 0] = 0

            # backward pass - second layer
            dW1 = np.dot(X.T, dhidden1)  # [3072 x 100]
            # dW1 += self.reg * W1
            db1 = np.sum(dhidden1, axis=0, keepdims=True)  # [1 x 100]

            # Perform parameter update ( SGD + Nesterov Momentum )

            # First layer
            v1_prev = v1  # back this up
            v1 = self.mu * v1 - self.learning_rate * dW1  # velocity update
            W1 += -self.mu * v1_prev + (1 + self.mu) * v1  # position update
            b1 += -self.learning_rate * db1

            # Second layer
            v2_prev = v2  # back this up
            v2 = self.mu * v2 - self.learning_rate * dW2  # velocity update
            W2 += -self.mu * v2_prev + (1 + self.mu) * v2  # position update
            b2 += -self.learning_rate * db2

            # Third layer
            v3_prev = v3  # back this up
            v3 = self.mu * v3 - self.learning_rate * dW3  # velocity update
            W3 += -self.mu * v3_prev + (1 + self.mu) * v3  # position update
            b3 += -self.learning_rate * db3

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

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
        W2 = nn['W2']
        b2 = nn['b2']
        W3 = nn['W3']
        b3 = nn['b3']

        # calculate first hidden layer activations
        hidden_layer1 = self.__forward(X, W1, b1)  # [50000 x 100]

        # calculate second hidden layer activations
        hidden_layer2 = self.__forward(hidden_layer1, W2, b2)  # [50000 x 100]

        # output neuron
        out = np.dot(hidden_layer2, W3) + b3  # [50000 x 10]

        predicted_labels = np.argmax(out, axis=1)  # [50000 x 1]

        # Calculate the accuracy
        acc = np.mean(predicted_labels == y)

        print("The final accuracy is : ", acc)

        return predicted_labels

    def __forward(self, X, W, b):
        """
         The forward pass for a neuron.
        :param X: the data.
        :param W: the weights.
        :param b: the bias.
        :return: return the firing data.
        """

        cell_body = np.dot(X, W) + b
        firing_rate = self.___activation(cell_body)

        return firing_rate

    def ___activation(self, X):
        """
        Activation function for a neuron.
        :param X: the data.
        :return: return the new data after apply the ReLU function
        """

        return np.maximum(0, X)  # ReLu

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

        return 0.5 * self.reg * np.sum(W * W)

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
    learn_data = 'result/NN1/cifar_10'

    D = 3072  # dimensionality
    K = 10  # number of classes

    # Neural Network
    nn = NeuralNetwork(D, K)

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
    util.save_predicted_labels('result/NN1/submission.csv', predicted_labels)
