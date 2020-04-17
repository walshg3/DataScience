import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1+np.exp(-1.0*x))

def feed_forward(features,w1,b1,w2,b2,w3,b3):
    HL1 = np.matmul(w1,features)
    HL1_with_bias = np.add(HL1,b1)
    HL1_with_bias_and_activation = np.maximum(HL1_with_bias,np.zeros((4,1)))

    HL2 = np.matmul(w2,HL1_with_bias_and_activation)
    HL2_with_bias = np.add(HL2,b2)
    HL2_with_bias_and_activation = np.maximum(HL2_with_bias,np.zeros((3,1)))
    
    targets_predicted = np.matmul(w3,HL2_with_bias_and_activation)
    targets_predicted = np.add(targets_predicted,b3)
    # Use sigmoid for the output activation
    targets_predicted = sigmoid(targets_predicted)
    return targets_predicted

def loss(features,w1,b1,w2,b2,w3,b3,targets_observed):
    targets_predicted = feed_forward(features,w1,b1,w2,b2,w3,b3)
    return np.sum((targets_predicted-targets_observed)**2)
#=====================================================
    
print('Starting ...')

## Set up training data
## Each row is a case
## Columns 0-4 are features
## Columns 5 & 6 are targets

features_and_targets = np.array( 
                                   [ [0, 0, 0, 0, 0, 0, 1],
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
                                     [1, 0, 0, 1, 1, 1, 0]  ]
                           , dtype=float)

# shuffle our cases
np.random.shuffle(features_and_targets)

# Need to transpose to get them as 5 X N matrices
features = np.transpose(features_and_targets[:,0:5])
targets_observed = np.transpose(features_and_targets[:,5:7])
number_of_features,number_of_cases = features.shape
print(number_of_features)
print(number_of_cases)
    
#Set initial weights and biases

np.random.seed(1)  
weights_1 = np.random.randn(4,5)
biases_1 = np.random.randn(4,number_of_cases)

weights_2 = np.random.randn(3,4)
biases_2 = np.random.randn(3,number_of_cases)

weights_3 = np.random.randn(2,3)
biases_3 = np.random.randn(2,number_of_cases)

# Targets_Predicted = feed_forward(features,weights_1,biases_1,
#                                       weights_2,biases_2,
#                                       weights_3,biases_3)

#print('Features : ',features)
#print(' Targets : ', targets_observed)
#print(' Targets predicted : ', Targets_Predicted)
learning_rate = 0.1
    
# Find slope functions using autograd

d_by_w1 = grad(loss,1)
d_by_b1 = grad(loss,2)
d_by_w2 = grad(loss,3)
d_by_b2 = grad(loss,4)
d_by_w3 = grad(loss,5)
d_by_b3 = grad(loss,6)

for epoch in range(10000):
    
    # At each iteration update weights and biases by subtracting 
    # learning_rate times slope 
    weights_1 -= learning_rate* d_by_w1(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed)
    biases_1 -= learning_rate*  d_by_b1(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed)  
    weights_2 -= learning_rate* d_by_w2(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed)
    biases_2 -= learning_rate*  d_by_b2(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed)   
    weights_3 -= learning_rate* d_by_w3(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed)
    biases_3 -= learning_rate*  d_by_b3(features,weights_1,biases_1,weights_2,biases_2,weights_3,biases_3,targets_observed) 

    # Print out the latest value of the loss 
    # We would expect this to go down with each iteration
    print(epoch,loss(features,weights_1,biases_1,
                        weights_2,biases_2,
                        weights_3,biases_3,targets_observed))
    
Targets_Predicted = feed_forward(features,weights_1,biases_1,
                                      weights_2,biases_2,
                                      weights_3,biases_3)

print('Features : ',features)
print(' Targets : ', targets_observed)
print(' Targets predicted : ', Targets_Predicted)  

N = 22
target1_predicted = Targets_Predicted[0,:]
target2_predicted = Targets_Predicted[1,:]
target1_observed = targets_observed[0,:]
target2_observed = targets_observed[1,:]

ind = np.arange(N) 
width = 0.35    
plt.subplot(2,1,1)
plt.bar(ind, target1_predicted, width, label='Predicted')
plt.bar(ind + width, target1_observed, width,label='Observed')
plt.ylabel('Targets 0 or 1')
plt.title('Closeness of predicted targets for 22 cases')
plt.xticks(ind + width / 2, ind)
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.bar(ind, target2_predicted, width, label='Predicted')
plt.bar(ind + width, target2_observed, width,label='Observed')
plt.ylabel('Targets 0 or 1')
plt.title('Closeness of predicted targets for 22 cases')
plt.xticks(ind + width / 2, ind)
plt.legend(loc='best')

plt.show()

# def feed_forward(features,w1,b1):
#     HL1 = np.matmul(w1,features)
#     HL1_with_bias = np.add(HL1,b1)
#     HL1_with_bias_and_activation = np.maximum(HL1_with_bias,0.0,HL1_with_bias)
#     return HL1_with_bias_and_activation