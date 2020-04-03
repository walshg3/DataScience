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
test = subset(df, indexes == FALSE)
# Remove the index variable from memory
rm(indexes)


###
## Start of the Logistic Regression
###

## Sigmoid Function 

sigmoid = function(wx){
  1 / (1 + exp(-wx))
}


