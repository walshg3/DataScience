## Greg Walsh
## Linear Regression 
## DSSA Machine Learning Fundamentals
## Spring 2020 

## Set wd
setwd('/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/Linear Regression')

## PseudoCode
## Read in the data
## Divide data into training and test data
## Train the model using the dataset:
##  Create X Matrix [1, xi1, xi2,...,xik] from all rows i of training data and for k variables
##  Create Y Matrix (vector)
## Solve for the β vector, 
## Make predictions using the test dataset:
##  Create X Matrix [1, xi1, xi2,...,xik] from all rows i of test data and for k variables
##  Compute Ŷ = βX
## Test the Model:
##  Compute the Sum of Squared Errors RSS 
##  Compute the Total Sum of Squares TSS 
##  Compute R2 = 1 – RSS / TSS

# Get the Data
df <- read.csv("TrainData_Group1.csv", header = TRUE)

# Separate the data into Test and Train Data. Train will be 80% of the dataset and Train will be the remaining 20% 
# Code taken from Slides
indexes <- sample(1:nrow(df), size=0.80*nrow(df)) 
# Set fifty of the data aside for the training
train <- df[indexes,]
# Set aside test data
test <- df[-indexes,]
# Remove the index variable from memory
rm(indexes)

## Start of the Linear Regression
LR <- function(df) {
  # create a column of 1's in the df
  # It should look like the Fit Model:
  # [1 x1 x2 x3 x4 x5 y] (We will remove the y later)
  df <- cbind(rep(1, nrow(df)), df)
  # We will need the length of df to create a subset later
  dflen <- length(df)
  # X Will be the columns including the 1's column added but without the Y column since that will be used to check out answers
  # We need to find what column has the name Y and get its subset with it removed
  X <- subset(df, select=-c(Y))
  # Convert X to a matrix for the LR Algorithm 
  X <- as.matrix(X)  
  #y is the subset of the column length
  y <- df[,dflen]   
  # Solve for the β vector
  # B_hat = (X^T * X)^-1 * X^T * y 
  # %*% is Matrix Multiplication
  # solve will  a %*% x = b for x, where b can be either a vector or a matrix.
  # The important thing here is a numeric or complex vector or matrix giving the right-hand side(s) of the linear system. If missing
  # b is taken to be an identity matrix and solve will return the inverse of a.
  # so its really 
  B_hat = solve(t(X) %*% X)  %*% t(X) %*% y
  return(B_hat)
}

# Ŷ = βX 
# Make Predictions using the test data
y_hat <- function(df, B_hat) {
  # Make a matrix with the X's without Y and Matrix Multiply it by B_hat (what gets returned from our LR)
  return (as.matrix(cbind(rep(1, nrow(df)), subset(df, select=-c(Y)))) %*% B_hat) 
}

#SSE TSS and R^2
ER <- function(y, y_hat, df) {
  # Sum of Squared Errors (y - Ŷi)^2 
  p <- subset(df, select=-c(Y))
  p <- ncol(p)
  SSE = sum((y - y_hat)^2)  
  # Total Sum of Squares (y - Ŷ_bar)^2
  TSS = sum((y - mean(y_hat))^2)
  # R^2 is 1 - SSE/TSS
  R2 <- 1 - SSE/TSS
  # My attempt at the Adjusted R2 
  adjR2 <- 1 - (SSE/(nrow(df)-p-1)/(TSS/nrow(df)-1))
  adjR2 <- mean(adjR2)
  return(list(R2 = R2, adjR2 = adjR2))
}

LRTest <- function(train, test){
  # Train the R's LR 
  LRTrain <- lm(data=train, formula = Y~.) 
  # Test the y_hat created by R's B_hat
  LRPredict = predict(LRTrain, newdata=test) 
  YPredict <- test$Y
  # plot the actual data vs R's predicted data 
  plot(x=YPredict, y=LRPredict, pch = "o", col='blue', main = "Actual VS Predicted by R")
  # Calculate the Errors
  error <- ER(test$Y, LRPredict, df)
  return(error$R2)
}

# Run my LR on the training data 
B_hat <- LR(train)

# Make predictions with the test data and my LR
# Note: every prediction we make will make the R2 lower 
LRPredict <- y_hat(test, B_hat)
test$Y
# plot The actual data vs my predicted data 
plot(x=test$Y, y=LRPredict, pch = "o", col='red', main = "Actual VS Predicted Using My LR")

# Calculate Error of my LR
error <- ER(test$Y, LRPredict, df)
# print the data to the user
print(paste("R^2 for my LR: ",error$R2, sep = " "))
print(paste("Adjusted R^2 for my LR: ",error$adjR2, sep = " "))
print(paste("R^2 for R's lm()", LRTest(train, test), sep = " "))

