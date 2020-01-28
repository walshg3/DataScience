# K-Nearest Neighbors code in R
# Coded Spring Semester 2018
# Dr. Clif Baldwin

# Write a K-Nearest Neighbors algorithm from scratch in R and then compare it to R's built-in knn() function.

# setwd("/media/clif/3763-3663/Stockton/Machine Learning")

####################################################################################
## NON_GRADED SECTION
# Make up some sample data for testing purposes
knn.df <- data.frame(x1 = sample(0:99, 30, replace=T),
                     x2 = sample(0:99, 30, replace=T),
					 y0 = sample(0:1, 30, replace=T))
# Instead of 0 and 1, make the response variable "g"ood or "b"ad
knn.df$y <- ifelse(knn.df$y0 == 1, "g","b")
# Now split the dataset into training and testing data
knn.df<- knn.df[sample(nrow(knn.df)),]
train.df <- knn.df[1:as.integer(0.7*(nrow(knn.df))),]
test.df <- knn.df[as.integer(0.7*(nrow(knn.df)) +1):(nrow(knn.df)),]
# The data is ready

# Set the value of K
K = 5

###################################################################################
## GRADED SECTION

# Algorithm
# 1. Calculate the Euclidean Distance (or whatever distance measurement you want) between the test data and the training data.
# 2. For each test data observation, arrange the calculated Euclidean distances in non-decreasing order.
# 3. Take the first k distances from this sorted list as the neighbors.
# 4. Determine the dominant response within the neighbors as the prediction for the test data observation.

# A function that computes the Euclidean Distance between a and b
# Euclidean Distance = square root of the sum of the squared differences
euclideanDist <- function(a, b){
  # a and b are vectors of unknown size
  distance = 0   # initialize distance
  for(i in c(1:(length(a)-1) )) # Not very efficient, but it works! Maybe you can do better?
  {
    distance = distance + (a[[i]]-b[[i]])^2  # summing the squared differences for each coordinate
  }
  distance = sqrt(distance)  # the square root of the summed squared differences
  return(distance)
}

# Using the training data, determine the k-nearest neighbors for the test data
knn_predict <- function(test_data, train_data, k_value){
  predictions <- vector("character",nrow(test_data))  #empty predictions vector
  
  #LOOP-1 loop through each test_data value and determine the training_data neighbors
  for(i in c(1:nrow(test_data))){   #looping over each record of test data
    euc_dist = vector("numeric",nrow(train_data))    # vector for Euclidean Distances
    euc_char = vector("character",nrow(train_data))  # Vector for corresponding response variables

    # 1. Calculate the Euclidean Distance between the test data point and training data
    #LOOP-2-looping over train data to get the Euclidean Distances from this test_data point
    for(j in c(1:nrow(train_data))){
      #  Save Euclidean Distances for each pair to euc_dist vector[]
      euc_dist[j] <- euclideanDist(test_data[i,], train_data[j,])

      # Save response variable of training data in euc_char
      euc_char[j] <- as.character(train_data[j,][[ncol(train_data)]])
    }

    distances <- data.frame(euc_char, euc_dist) #distances dataframe created with euc_char & euc_dist

    # 2. For each test observation, arrange the calculated Euclidean distances in non-decreasing order
    distances <- distances[order(distances$euc_dist),]       #sorting distances to get top K neighbors
    
    # 3. Take the first K distances from the sorted list as the neighbors
    distances <- distances[1:k_value,]               #distances dataframe with top K neighbors

    # Count the classes of neigbhors in distances.
    good <- length(which(distances$euc_char == "g"))
    bad <- length(which(distances$euc_char != "g"))

    # 4. Determine the dominant response within the neighbors as the prediction for the test data observation
    if(good > bad){     #if majority of neighbors are good then put "g" in predictions vector
      predictions[i] <- "g"
    }
    else {   ## if(good <= bad)
      predictions[i] <- "b"    #if non-majority of neighbors are good then put "b" in predictions vector
    } # end if-else

  } # End for(i in c(1:nrow(test_data)))
  return(predictions) #return predictions vector, which are the predictions for the test data
}

# A function to compute the accuracy statistic - not very efficient
accuracy1 <- function(test_data){
  correct = 0 #initialize the counter of correct predictions
  # loop through test data and mark correct when the prediction matches
  for(i in c(1:nrow(test_data))){
    if(test_data[i,ncol(test_data)-1] == test_data[i,ncol(test_data)]){
      correct = correct+1 # Increment counter when test data response matches predicted response
    } # end if predictions match
  } # end for loop
  # Compute percentage correct
  accu = correct/nrow(test_data) * 100
  return(accu)
} # Do not use this function - too inefficient! Use the following function instead.

# A function that utilizes vectorization to compute the accuracy statistic
accuracy <- function(test_data){
  # Count correct when the prediction matches the test data
  correct = length(which(test_data[,ncol(test_data)-1] == test_data[,ncol(test_data)]))
  
  # Compute the percentage correct
  accu = correct/nrow(test_data) * 100
  return(accu)
}

# K-Nearest Neighbors
predictions <- knn_predict(test.df, train.df, K) #calling knn_predict()

### Save the predicted values with the test data
test.df[,ncol(test.df)+1] <- predictions #Adding predictions in test data as a new column

### Compute and Print the accuracy statistic
print(accuracy(test.df))


### Display a graph of the data
library("ggplot2")  # ggplot2 is an acceptable library since it is not related to the algorithm!
# Graph the training data and use color to indicate the response variable
ggplot(train.df, aes(x1,x2)) + geom_point(colour=ifelse(train.df$y0>0,"blue","red"))

# Add the test data. Use green when the prediction matches and orange for misses
ggplot(train.df, aes(x1,x2)) + geom_point(colour=ifelse(train.df$y0>0,"blue","red")) +
  geom_point(data=test.df, aes(x1, x2), colour=ifelse(test.df$y==test.df$V5,"green","orange")) +
  labs(title = "k-NN Test Data", subtitle = "Training = Blue & Red; Green = match; Orange = miss")
  
##################################################
# Now compare the results to the R function knn()
# We need to get the appropriate library for knn()
library(class) # knn() is in the library "class"

# Use the same data again but I am storing it with a different name
train <- knn.df[1:as.integer(0.7*(nrow(knn.df))),]
test <- knn.df[as.integer(0.7*(nrow(knn.df)) +1):(nrow(knn.df)),]

# Run knn() and save the predictions to the test dataframe
test[,ncol(test)+1] <- knn(train[,c(1,2)], test[,c(1,2)], train[,3],k=5, prob=FALSE)

# knn() returns the prediction as 0 or 1, but I want "b" or "g"
# If 0, change to "b", else change to "g"
test$V5 <- as.character(ifelse(test$V5 == 0,"b","g"))

# Print the accuracy results of knn()
print(accuracy(test))
