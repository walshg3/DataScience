## Greg Walsh
## Machine Learning Fundamentals 
## KMeans Implementation
## Spring 2020 

## Set Working Directory 
setwd("/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/KMeans")
## Imports Libraries
library(tidyverse)

## Import Data
KMeansData_Group1 <- read_csv("KMeansData_Group1.csv")
names(KMeansData_Group1) [1] <- "x"
names(KMeansData_Group1) [2] <- "y"
x = as.matrix(KMeansData_Group1[1:2])
KMeansData_Group2 <- read_csv("KMeansData_Group2.csv")



## K Means Clustering

### Algorithm 
#Step 1: Choose groups in the feature plan randomly
#Step 2: Minimize the distance between the cluster center and the different observations (centroid). It results in groups with observations
#Step 3: Shift the initial centroid to the mean of the coordinates within a group.
#Step 4: Minimize the distance according to the new centroids. New boundaries are created. Thus, observations will move from one group to another
#Repeat until no observation changes groups
###

### PSEUDOCODE
# 1. Create k points for starting centroids
# 2. While any point has changed cluster assignment
#    a. For every point in our dataset:
#         For every centroid:
#           Calculate the distance between the centroid and point
#         Assign the point to the cluster with the lowest distance
#    b. For every cluster calculate the mean of the points in that cluster
#         Assign the centroid to the mean
###


## From Prof
## Create k points for starting centroids
ks <- 2:25 


# Initialize a vector to hold each K's Total Within-Sum of Squares
wss = vector(mode = "numeric", length = length(ks))

for(i in ks) {
  kfish <- kmeans(KMeansData_Group1, centers = i)
  wss[i-1] = kfish$tot.withinss # Save each Total Sum of Squares to the vector
} # Now the K Means has been computed for K = 2, 3, 4, ..., 25


plot(ks, wss, type="b", xlab="Number of Clusters",
     pch=20, cex=2)

x = as.matrix(KMeansData_Group1[1:2])

# R Build in Euclidean Distance
stats::dist(KMeansData_Group1[1:2], method= "euclidean")
distance <- dist(KMeansData_Group1[1:2], method = "euclidean")

# A function for calculating the distance between centers and the rest of the dots
euclid <- function(points1, points2) {
  distanceMatrix <- matrix(NA, nrow=dim(points1)[1], ncol=dim(points2)[1])
  for(i in 1:nrow(points2)) {
    distanceMatrix[,i] <- sqrt(rowSums(t(t(points1)-points2[i,])^2))
  }
  distanceMatrix
}


# A method function
K_means <- function(x, centers, euclid, nItter) {
  clusterHistory <- vector(nItter, mode="list")
  centerHistory <- vector(nItter, mode="list")
  
  for(i in 1:nItter) {
    distsToCenters <- euclid(x, centers)
    clusters <- apply(distsToCenters, 1, which.min)
    centers <- apply(x, 2, tapply, clusters, mean)
    # Saving history
    clusterHistory[[i]] <- clusters
    centerHistory[[i]] <- centers
  }
  
  structure(list(clusters = clusterHistory, centers = centerHistory))
  
}
centroids = KMeansData_Group1[sample.int(nrow(KMeansData_Group1), 5), ]

test <- K_means(as.matrix(KMeansData_Group1), centroids, euclid, 5)


myKmeans <- function(dataset, k, max){
  # List of past centroids for graphing later
  datacollength <- length(dataset[1,])
  datarowlength <- nrow(dataset)
  #Create Centroids
  #from the dataset get total rows and grab k random rows 
  centroids = dataset[sample.int(nrow(dataset), k), ]
  
  
}

euclid <- function(points1, points2) {
  distanceMatrix <- matrix(NA, nrow=dim(points1)[1], ncol=dim(points2)[1])
  for(i in 1:nrow(points2)) {
    distanceMatrix[,i] <- sqrt(rowSums(t(t(points1)-points2[i,])^2))
  }
  distanceMatrix
}

#cluster_dist<-function(X,cluster_centers){ distMatrix = matrix(NA, nrow= nrow(X), ncol = nrow(cluster_centers)) for(j in 1:nrow(cluster_centers)){ for(i in 1:nrow(X)){ distMatrix[i,j]<-dist(rbind(X[i,],cluster_centers[j,])) } } distMatrix }

K_means <- function(x, centers, distFun, nItter) {
  clusterHistory <- vector(nItter, mode="list")
  centerHistory <- vector(nItter, mode="list")
  
  for(i in 1:nItter) {
    distsToCenters <- distFun(x, centers)
    clusters <- apply(distsToCenters, 1, which.min)
    centers <- apply(x, 2, tapply, clusters, mean)
    # Saving history
    clusterHistory[[i]] <- clusters
    centerHistory[[i]] <- centers
  }
  
  list(clusters=clusterHistory, centers=centerHistory)
}

ktest=as.matrix(KMeansData_Group1)
centers <- ktest[sample(nrow(ktest), 5),]
res <- K_means(ktest, centers, euclid, 10)

plot(length(res$clusters), res$clusters)

