setwd("/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/KMeans")

# January 31, 2020

library(tidyverse)

# Source of the dataset
#https://www.kaggle.com/new-york-state/nys-fish-stocking-lists-actual-beginning-2011
#https://www.dec.ny.gov/outdoor/30467.html 
#https://data.ny.gov/Recreation/Fish-Stocking-Lists-Actual-Beginning-2011/e52k-ymww

# Dataset Metadata (description of the attributes)
#County - NYS county where stocking occurred.
#Waterbody - Name of waterbody in which the fish were stocked.
#Town - Town where stocking occurred.
#Month - Month in which the fish were stocked.
#Number - Number of fish stocked.
#Species - Species of fish stocked.
#Size (Inches) - Size range of fish stocked.

# A Function to normalize data
normalize <- function(x) { (x - min(x)) / (max(x) - min(x))}

# A function to compute the accuracy
# Number correct divided by total number of predictions, as saved in a table
accuracy <- function(x) { sum(diag(x) / (sum(rowSums(x))))}

#### Read and clean the data ###
# Read in the dataset
fish <- read_csv("fish-stocking-lists-actual-beginning-2011.csv")

# Only keep certain attributes (Number, Length, and Species)
fish <- fish[,c(6:8)]

# How many rows in the dataset
nrow(fish)

# How many rows are complete?
sum(complete.cases(fish))

# So six observations are missing at least one of Number, Length, and Species
# Only 6 incomplete though! 
# Just delete them because I doubt such a small amount will bias the dataset.
fish <- fish[complete.cases(fish),]

# Although not missing, so observations have length 0.
# No fish has length 0!
# So delete the observations when the size is 0! There are only 5.
fish <- fish[fish$`Size (Inches)` > 0,] #Save observation when the fish as a length

# Column 2 is the response variable.
# We would never normalize a response variable
# (Remember we would not have a response variable for unsupervised learning)
# Normalize the predictors though
fish_norm <- as.data.frame(lapply(fish[,c(1,3)], normalize))

# Species is a "response variable" 
# We would not have a response variable when using K Means Clustering,
# but we will need it for the supervised learning K-Nearest Neighbors algorithm.
fish_norm$Species <- factor(fish$Species) # Add Species to normalized dataset

# Dataset seems cleaned and ready for analysis.

### K Means Clustering ###
# My research question for the data without labels (pretend Species is not an attribute)
# I want to determine groups based on the size and number

ks <- 2:25 # different K's 
# I do not want K=1 because that would mean everything is part of one group,
# which is not interesting at all!

# Initialize a vector to hold each K's Total Within-Sum of Squares
wss = vector(mode = "numeric", length = length(ks))

# Compute the K Means Clusters for clusters k = 2 to 25
# kmeans() is automatically loaded with R
for(i in ks) {
  kfish <- kmeans(fish_norm[,c(1,2)], centers = i)
  wss[i-1] = kfish$tot.withinss # Save each Total Sum of Squares to the vector
} # Now the K Means has been computed for K = 2, 3, 4, ..., 25


# Print the Elbow Curve
# The x-axis are the ks (i.e. K=2, 3, 4, 5, ..., 25)
# The y-axis are the Total Sum of Square errors
plot(ks, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="The Elbow Curve of Clusters for Fish",
     pch=20, cex=2)


# Looks like around 14, but I know there are 20 species 
# (remember, I should not "know" that)
kfish <- kmeans(fish_norm[,c(1,2)], centers = 20)

# Graph the clusters
ggplot(fish_norm,aes(x=Size..Inches.,y=Number,color=as.factor(kfish$cluster))) +
  geom_point() + labs(color = "Cluster", title = "Clusters Centers")

# Graph the species, which we should not be able to do for unsupervised learning
# But for demonstration purposes, we have the response variable
ggplot(fish_norm,aes(x=Size..Inches.,y=Number,color=as.factor(Species))) +
  geom_point() + labs(color = "Species")

# Well those two graphs are not very helpful
# Let us explore the data a bit
fish_norm[fish_norm$Size..Inches. > 0.7,] # Remember the data is normalized
# The size of the Rainbow Trout are messing up the graph!

fish_norm[fish_norm$Number > 0.1,] # Remember the data is normalized
# Or is it the count of Walleyes?

nrow(fish_norm[fish_norm$Number > 0.01,]) # Remember the data is normalized
# There are 48 fish out of 21569 total!
nrow(fish_norm[fish_norm$Species=="Walleye",])
# And 366 Walleye

# Weed out the 48 Walleyes, which might be significant enough to bias the data!!
fish_bias <- fish_norm[fish_norm$Number <= 0.01,]
cluster <- kfish$cluster[fish_norm$Number <= 0.01]
# The data is now potentially biased!!! 
# Only use this subset for the special graph and then delete it!

ggplot(fish_bias,aes(x=Size..Inches.,y=Number,color=as.factor(cluster))) +
  geom_point() + labs(color = "Cluster", title = "Clusters with biased Walleye")

ggplot(fish_bias,aes(x=Size..Inches.,y=Number,color=as.factor(Species))) +
  geom_point() + labs(color = "Species (with biased Walleye")
# A little easier to see
# The groups look similar to the different species.
# IF I did not know how the species differed, 
#    this grouping helps me determine the difference in species.
  
# We are finished with K Means Clustering here.
# Clean up the memory a bit
rm(kfish, fish_bias, cluster, i, ks, wss)

### K Nearest Neighbors ###

# Species is the response variable, which we should not have known for K Means Clustering
# But a response variable is necessary for K Nearest Neighbor

# My research question for the data with labels
# I want to predict the Species based on the size and number

# The knn() function is not automatically loaded in R
# We can load it by calling the library class
library(class) # Needed for knn()

# Whenever we are doing Supervised Learning,
#  we want to test the predictions.
# To do so, we save some of the data for testing.
# Sometimes I save 25% for testing, but I think 10% is adequate here.
# Split the dataset into training and testing data
fish_norm <- fish_norm[sample(nrow(fish_norm)),] # shuffle up (randomize) the data
train <- fish_norm[1:as.integer(0.9*(nrow(fish_norm))),] # Save 90% for training
# Reserve 10% of the observations to test the KNN model
test <- fish_norm[as.integer(0.9*(nrow(fish_norm)) +1):(nrow(fish_norm)),]

# Run the knn() function.
# Provide the training data, testing data, and the response variables of the training data
# I am guessing that K=7 (similarity to 7 neighbors) will work well
testYield <- knn(train[,c(1:2)], test[,c(1:2)], cl=train[,3], k=7)
# Create a table to compare the predictions for the testing data
#   with the actual responses of the testing data
tab <- table(testYield, test[,3])
# Run the accuracy function I wrote to compute the percentage that the prediction got correct
accuracy(tab)
# ~75% is much better than "chance"

# My model works, but maybe I can do better.
# Maybe my guess of K=7 is not the best.
# So I will change my guess to k=9 neighbors
testYield <- knn(train[,c(1:2)], test[,c(1:2)], cl=train[,3], k=9)
tab <- table(testYield, test[,3])
accuracy(tab)
# Also around 75%
# K=9 is not really different than K=7.
# I should continue trying different K=??
# I could also see if I change how Number and Length are weighted.
# Currently they are both normalized to 0 - 1.
# I could change the weighting of one of them and try again.
# Maybe I suspect Length is more important than Number.
# I could keep Number normalized from 0 - 1 but 
#   change Length to "normalized" from 0 - 2??
# Then Length would have twice the weight of Number.
# Making these changes is called "tuning the model."
# It takes some time, effort, and some experience (or just plain luck).

