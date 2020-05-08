## Greg Walsh
## DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS
## Logistic Regression
## Spring 2020 

## PseudoCode
# Read the data
# Divide data into training and test data
# Determine the number of iterations, n
# Train the model using the training dataset
## Create a matrix X from the training data with k features and m observations
## Create the Y matrix (actually a vector) 
## Create a matrix (actually vector), W, of k zeroes (e.g. [0, 0, ., 0] )
## Loop for n iterations
### Compute the sigmoid result, g(x) = 1 / (1 + e^-(WX))
### Compute the gradient = (1/m) * X(g(x) - Y)
### W = W - (0.001 * gradient)
# Make predictions using the test dataset
## Computer Yhat = 1 / 1 + e^-(W^T*X)
# Test the model 

## Import statements and Work Directory
library(ggplot2)
library(caTools)
setwd("C:/Users/Gregwalsh96/github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/LogisticRegression")

###
## Import Data
###
df <- read.csv("LogisticData_1.csv", header = TRUE)
## Split Data into Testing and Training
## Using csTools Like describe in the powerpoint slides 
set.seed(8675309) #https://www.youtube.com/watch?v=6WTdTwcmxyo
# Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
indexes = sample.split(df$Y, SplitRatio = 0.75)
## Split data into test and train using caTools
train = subset(df, indexes == TRUE)
test  = subset(df, indexes == FALSE)
# Remove the index variable from memory
rm(indexes)

###
## Start of the Logistic Regression
###

## Sigmoid Function 

sigmoid = function(x){
  1 / (1 + exp(-x))
}
head(train[1:2])
# insert x1 and x2 values in a matrix
x <- as.matrix(train[1:2])
# add an extra row of all 1's for our matrix multiplaction later
x <- cbind(x, intercept=1)
# get y values in a matrix 
y <- as.matrix(train[3])


# we take in our x and y values from above
GWLogistic=function(x,y)
{
  # create an iter variable
  # create a boolean variable to determine if we can stop
  # create max iterations n 
  i = 0
  n = 100
  converged = FALSE
  # tolerance 
  alpha = 0.001
  # create coefficients variable 
  coefficients = 1
  # set matrix weights equal to 1 to start and set dimension of the matrix 
  coefficients = matrix(coefficients, dim(x)[2])
  
  # update weights
  # loop will stop when converged or iterations are reached  
  while (i < n & !converged){
    # count iter
    i=i+1
    # run the x values and coefficients through the Sigmoid function
    predictGW <- sigmoid(x %*% coefficients)
    # set the diagonals of the matrix to 0 
    # print(diag(predictGW[,1]))
    predictGW_diag = diag(predictGW[,1])
    # update weights
    # solve() will  find the co-effecients in the equation 
    # and also to find the inverse of given matrix.
    # we use the coefficients the solve() function finds to update
    # the current coefficients 
    coefficients = coefficients + solve(t(x) %*% predictGW_diag %*% x) %*% 
      t(x) %*% (y - predictGW)
    # compute mean squared error to check completion 
    meansquarederror = mean((y - sigmoid(x %*% coefficients))^2)
    # if mean squared error is less than the alpha we can stop 
    if (meansquarederror < alpha){
      converged = TRUE
    }
  }
  # Print the values 
  pred = sigmoid(x %*% coefficients)
  resid = pred - y
  meansquared = mean(resid^2)
  confmatrix = table(Actual = y, Predicted = pred > 0.5)
  accuracy = (sum(diag(confmatrix))) / (sum(confmatrix))
  print('------------------------------')
  print('My Model')
  print('------------------------------')
  print('My Model coefficients')
  print(coefficients)
  print('------------------------------')
  print('My Model Accuracy')
  print(accuracy)
  print('------------------------------')
  print('My Model Confusion Matrix')
  print(confmatrix)
  print('------------------------------')
}
# Run my Model
training_results <- GWLogistic(x,y)

# Test with R's Logistic Regression

### Y is our response variable and X1 and X2 are the predictor variables.
### We are using Generalized linear models glm(), 
### with family link as binomial 
model <-  glm(Y~., train, family="binomial")

# lets take a look at the model 
# summary(model)

# The coefficients of the model are the intercepts
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  3.40205    0.24038  14.153  < 2e-16
#   X1           0.08504    0.15154   0.561    0.575    
# X2           1.38121    0.18085   7.637 2.22e-14


## predict() function predicts the Y variable from test data using 
## X1 and X2 variables.
## This is built into R
results <-  predict(model, test, type="response")
str(results)

print('------------------------------')
print('R Model')
print('------------------------------')
confmatrix <- table(Actual = test$Y, Predicted = results > 0.5)
print('R Confusion Matrix')
print(confmatrix)


accuracy <- confmatrix
accuracy <- (accuracy[1] + accuracy[4]) / sum(accuracy)
print(paste('R model accuracy ', accuracy))


### Output ###
# "------------------------------"
#  "My Model"
# "------------------------------"
#  "My Model coefficients"
# Y
# X1        0.08050428
# X2        1.28713460
# intercept 3.26684580
# "------------------------------"
#  "My Model Accuracy"
#  0.928
# "------------------------------"
#  "My Model Confusion Matrix"
#         Predicted
# Actual FALSE TRUE
#      0     2   52
#      1     2  694
# "------------------------------"
# "------------------------------"
# "R Model"
# "------------------------------"
#  "R Confusion Matrix"
#         Predicted
# Actual FALSE TRUE
#      0     5   13
#      1     0  232
# "R model accuracy  0.948"
