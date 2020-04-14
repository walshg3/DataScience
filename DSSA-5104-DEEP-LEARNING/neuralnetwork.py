'''
Greg Walsh
Feed Forward
DSSA Deep Learning Week 6
Spring 2020


Submit your code that takes 5 feature values and computes two targets
based on random values for all weights and biases and gives the
user the option of choosing either a RELU or a sigmoid function.
The computation of the targets should be in its own Python
function and should accept the five features and all your weights and
biases as input and should output the targets.

'''
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

# Get User Input to use RELU or SIGMOID

while True:
    select = input("Please enter 'r' or 's' or 'c' for RELU or Sigmoid or Combo of both respectively \n")
    if select in ["r", "s", "c"]:
        break


def feed_forward(features, w1, b1, w2, b2, w3, b3, select):
    # Sigmoid
    # Mine
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    # def sigmoid(x):
    #     return 1.0/(1+np.exp(-1.0*x))
    # RELU

    def relu(x):
        return np.maximum(x, 0)
    # Run Relu or Sigmoid depending on the output
    # Note: The activation function gets ran on every layer
    if select == "s":
        # sigmoid
        # hidden layer 1
        hl1 = (np.matmul(w1, features))
        hl1_bias = np.add(hl1, b1)
        hl1_act = sigmoid(hl1_bias)
        # hidden layer 2
        hl2 = (np.matmul(w2, hl1_act))
        hl2_bias = np.add(hl2, b2)
        hl2_act = sigmoid(hl2_bias)
        # output layer
        output = (np.matmul(w3, hl2_act))
        targets_predicted = np.add(output, b3)
        targets_predicted = sigmoid(targets_predicted)
    elif select == "r":
        # relu
        # hidden layer 1
        hl1 = (np.matmul(w1, features))
        hl1_bias = np.add(hl1, b1)
        hl1_act = relu(hl1_bias)
        # hl1_act = np.maximum(hl1_bias, np.zeros((4, 1)))
        # hidden layer 2
        hl2 = (np.matmul(w2, hl1_act))
        hl2_bias = np.add(hl2, b2)
        hl2_act = relu(hl2_bias)
        # hl2_act = np.maximum(hl2_bias, np.zeros((3, 1)))
        # output layer
        output = (np.matmul(w3, hl2_act))
        targets_predicted = np.add(output, b3)
        # targets_predicted = sigmoid(targets_predicted)
        targets_predicted = relu(targets_predicted)
        # targets_predicted = np.maximum(targets_predicted, np.zeros((3, 1)))
    elif select == "c":
        # relu
        # hidden layer 1
        hl1 = (np.matmul(w1, features))
        hl1_bias = np.add(hl1, b1)
        hl1_act = relu(hl1_bias)
        # hl1_act = np.maximum(hl1_bias, np.zeros((4, 1)))
        # hidden layer 2
        hl2 = (np.matmul(w2, hl1_act))
        hl2_bias = np.add(hl2, b2)
        hl2_act = relu(hl2_bias)
        # hl2_act = np.maximum(hl2_bias, np.zeros((3, 1)))
        # output layer
        output = (np.matmul(w3, hl2_act))
        targets_predicted = np.add(output, b3)
        # targets_predicted = sigmoid(targets_predicted)
        targets_predicted = sigmoid(targets_predicted)
        # targets_predicted = np.maximum(targets_predicted, np.zeros((3, 1)))
    return targets_predicted


def loss(features, w1, b1, w2, b2, w3, b3, targets_observed, select):
    '''
    You will need to add your code here that propagates features
    through the network to get predicted_targets
    based on matrix multiplication and addition (with weights and biases)
    and employing activation functions. I have used w1 to represent the weights
    matrix for the transition from the input layer to the first hidden layer
    and b1 to represent the biases added at the first hidden layer.
    I have used w2 to represent the weights
    matrix for the transition from the first hidden layer to the second hidden
    layer and b2 to represent the biases added at the second hidden layer.
    I have used w3 to represent the weights
    matrix for the transition from the second hidden layer to the output
    layer and b3 to represent the biases added at the output layer
    Usage: Calculate the sum of square residuals of the feed forward function
    '''
    Targets_Predicted = feed_forward(features, w1, b1, w2, b2, w3, b3, select)
    return np.sum((Targets_Predicted - targets_observed) ** 2)


print('You selected: ' + select)
print('Engines Starting ...')
print('Hold Tight Running Epochs')

# Set up training datam
# Each row is a case
# Columns 0-4 are features
# Columns 5 & 6 are targets

features_and_targets = np.array(
                                   [[0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 1, 1, 0, 1],
                                    [0, 0, 1, 1, 1, 0, 1],
                                    [0, 1, 1, 1, 1, 0, 1],
                                    [1, 1, 1, 1, 0, 0, 1],
                                    [1, 1, 1, 0, 0, 0, 1],
                                    [1, 1, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 1, 0, 0, 1],
                                    [1, 0, 1, 1, 0, 0, 1],
                                    [1, 1, 0, 1, 0, 0, 1],
                                    [0, 1, 0, 1, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 1, 1, 1, 1, 0],
                                    [1, 1, 0, 1, 1, 1, 0],
                                    [1, 0, 1, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 1, 1, 0],
                                    [1, 1, 0, 0, 1, 1, 0],
                                    [1, 1, 1, 0, 1, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 0],
                                    [1, 0, 0, 1, 1, 1, 0]], dtype=float)

# shuffle our cases
np.random.shuffle(features_and_targets)

# Transpose Matrix for mat mul in feed forward
features = np.transpose(features_and_targets[:, 0:5])
targets_observed = np.transpose(features_and_targets[:, 5:7])
number_of_features, number_of_cases = features.shape
print('Number of Features:', number_of_features)
print('Number of Cases:', number_of_cases)

# Set initial weights and biases
# use a seed so others can replicate results
losses = []

# np.random.seed(8675)

weights_1 = np.random.rand(4, 5)
biases_1 = np.random.rand(4, number_of_cases)

weights_2 = np.random.rand(3, 4)
biases_2 = np.random.rand(3, number_of_cases)

weights_3 = np.random.rand(2, 3)
biases_3 = np.random.rand(2, number_of_cases)

# Set our learning rate
lr = 0.001

# find slope
# If you have created a loss function this way then computing the functions
# for the gradients of this loss function with respect to all weights
# and biases is easy with autograd You can then update the weights
# and biases with gradient descent using a learning rate.

# grad loss, variable you want to look at

d_loss_by_d_w1 = grad(loss, 1)  # w1
d_loss_by_d_b1 = grad(loss, 2)  # b1
d_loss_by_d_w2 = grad(loss, 3)  # w2
d_loss_by_d_b2 = grad(loss, 4)  # b2
d_loss_by_d_w3 = grad(loss, 5)  # w3
d_loss_by_d_b3 = grad(loss, 6)  # b3

# Create epoch for our back tracking.
# Backpropagate to calculate the gradient for each weight
epochs = 10000

for epoch in range(epochs):

    weights_1 -= lr * d_loss_by_d_w1(features, weights_1, biases_1, weights_2,
                                     biases_2, weights_3, biases_3,
                                     targets_observed, select)

    biases_1 -= lr * d_loss_by_d_b1(features, weights_1, biases_1, weights_2,
                                    biases_2, weights_3, biases_3,
                                    targets_observed, select)

    weights_2 -= lr * d_loss_by_d_w2(features, weights_1, biases_1, weights_2,
                                     biases_2, weights_3, biases_3,
                                     targets_observed, select)

    biases_2 -= lr * d_loss_by_d_b2(features, weights_1, biases_1, weights_2,
                                    biases_2, weights_3, biases_3,
                                    targets_observed, select)

    weights_3 -= lr * d_loss_by_d_w3(features, weights_1, biases_1, weights_2,
                                     biases_2, weights_3, biases_3,
                                     targets_observed, select)

    biases_3 -= lr * d_loss_by_d_b3(features, weights_1, biases_1, weights_2,
                                    biases_2, weights_3, biases_3,
                                    targets_observed, select)
    losses.append(loss(features, weights_1, biases_1, weights_2,
                       biases_2, weights_3, biases_3,
                       targets_observed, select))

    # used for testing purposes. If you want to see how the
    # loss backpropagate is calculating a lower gradient run this
    # print(epoch, loss(features, weights_1, biases_1, weights_2, biases_2,
    #                   weights_3, biases_3, targets_observed, select))


Targets_Predicted = feed_forward(features, weights_1, biases_1, weights_2,
                                 biases_2, weights_3, biases_3, select)




print('Features : \n', features)
print(' Targets : \n', targets_observed)
print(' Targets predicted : \n', Targets_Predicted)

# plt.semilogy(losses)  # make a plot with log scale on the y axis
plt.plot(losses)
plt.text(3000, 3, 'The NNs performance at learning rate 0.001  ')  # Add text on plot
plt.xlabel('Number of epochs')  # Add label name
plt.title('Learning Curve using Sigmoid Activation Function ')  # Add title name
plt.ylabel('Target Values Observed')  # Add label name
plt.show()  # Plot graph
plt.savefig('lr001c.png')  # Save figure


'''
Code to show observed vs predicted
'''
N = 22
target1_predicted = Targets_Predicted[0, ]
target2_predicted = Targets_Predicted[1, :]
target1_observed = targets_observed[0, :]
target2_observed = targets_observed[1, :]
ind = np.arange(N)
width = 0.35
plt.subplot(2, 1, 1)
plt.bar(ind, target1_predicted, width, label='Predicted')
plt.bar(ind + width, target1_observed, width, label='Observed')
plt.ylabel('Targets 0 or 1')
plt.title('Closeness of predicted targets for 22 cases')
plt.xticks(ind + width / 2, ind)
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.bar(ind, target2_predicted, width, label='Predicted')
plt.bar(ind + width, target2_observed, width, label='Observed')
plt.ylabel('Targets 0 or 1')
plt.title('Closeness of predicted targets for 22 cases')
plt.xticks(ind + width / 2, ind)
plt.legend(loc='best')

plt.show()
