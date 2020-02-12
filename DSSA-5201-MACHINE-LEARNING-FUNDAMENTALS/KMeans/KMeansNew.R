## Set Working Directory 
setwd("/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/KMeans")
## Imports Libraries
library(tidyverse)
library(class)


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

### How to Determine when to stop
# Stopping can be done by running x iterations 
#   (+ easy to implement [run for loop x times])
#   (- extra compuatation time, why run 20 times when it completes in 5?)
##
# Check if the Centroids (Centers) have moved since the last iteration
#   (+ only run computation times for what is needed)
#   (- will require extra space to keep the old centroids and will require extra compute to determine if centroid has moved)
#   For this we will need a stop criteria. A min value value the Centroids will have to move in order to be in the threshhold 
###

## Import Data

## From reading online using Kmeans with a data structure in the form of a matrix makes manipulating vectors easier
KMeansData_Group1 <- as.matrix(read.csv("KMeansData_Group1.csv", header = FALSE)) 

kmeansnew = function(data, K, stop_crit=10e-3){
  ## Init Variables
  # 1. Create k points for starting centroids
  # Centroids
  centroids = data[sample.int(nrow(data),K),]
  # Cluster (assigned center points)
  # rep will replicate the data https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/rep
  cluster = rep(0,nrow(data))
  # When to stop
  current_stop_crit = 1000
  ## Variables to help determine convergence
  
  converged = FALSE
  #Counter variable
  iter = 1
  # Determine if centroids have stopped moving
  while (current_stop_crit >= stop_crit && converged == FALSE) {
    iter = iter + 1 
    sapply(current_stop_crit, function(x) if (x <= stop_crit) {converged = TRUE})
    old_centroids = centroids
    # Iter over the data
    for (i in 1:nrow(data)){
      # Set High beginning min distance
      min_dist=10000000
      # Iter over the Centroids
      for (centroid in 1:nrow(centroids)){
        # calculate the euclidian distance
        distance_to_centroid = sum((centroids[centroid,]-data[i,])^2)
        ## KEep Total Distance 
        # determine closest centroid to the min_dist
        if (distance_to_centroid <= min_dist){
          cluster[i] = centroid
          min_dist = distance_to_centroid
        }
      }
    }
    # Update Centroids 
    # for each centroid
    for (i in 1:nrow(centroids)){
      # determine new centroid based on means of the clusters
      centroids[i,] = apply(data[cluster == i,], 2, mean)
    }
    # Recaluclate stop critiera
    # See if the centroids stopped moving
    #current_stop_crit = mean((old_centroids - centroids) ^2)
    current_stop_crit = sapply((old_centroids-centroids)^2, mean, na.rm = TRUE)
    #print(paste0("current_stop_crit: ", current_stop_crit))
  }
  centroids = data.frame(centroids, cluster=1:K)
  return(list(data = data.frame(data,cluster), centroids = centroids))
}

kmeansplotnew <- function(K){
  res1=kmeansnew(KMeansData_Group1,K)
  #print(paste0('sumofsqu: ', res1$sumofsqu))
  #res1$centroids$cluster=1:K
  res1$data$isCentroid=F
  res1$centroids$isCentroid=T
  #names(res1$data) <- names(res1$centroids) 
  data_plot1=rbind(res1$centroids,res1$data)
  ggplot(data_plot1,aes(x=V1,y=V2,color=as.factor(cluster),size=isCentroid,alpha=isCentroid))+geom_point()
}

#K <- 6
#res <- kmeansnew(KMeansData_Group1, K)
#res$data$isCentroid=F
#res$centroids$isCentroid=T
#data_plot1=rbind(res$centroids,res$data)

kmeansplotnew(5)

computesumsqr <- function(kmdata,K){
wss = 0.0
df <- data.frame()
for (i in 1:K) {
  for (j in 1:nrow(kmdata$data)) {
    if(kmdata$data$cluster[j] == kmdata$centroids$cluster[i]){
      df <- data.frame(kmdata$data$V1[j],kmdata$centroids$V1[i], kmdata$data$V2[j],kmdata$centroids$V2[i] )
      wss = wss + (kmdata$data$V1[j] - kmdata$centroids$V1[i])^2 + (kmdata$data$V2[j] - kmdata$centroids$V2[i])^2
    }
  }
}

  return(wss)
  #print(wss)
  rm(i,j)
}

ks <- 2:15


wss <- vector("numeric", length(ks)-1)
for (i in ks){
  print(paste0('K is: ', i))
  res <- kmeansnew(KMeansData_Group1, i)
  wss[i-1] = computesumsqr(res, i)
  print(wss[i-1])
}

plot(ks, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="The Elbow Curve of Clusters",
     pch=20, cex=2)













rk <- kmeans(KMeansData_Group1, centers = 5)
KMeansData_Group1$cluster <- as.character(rk$cluster)
ggplot() + geom_point(data = KMeansData_Group1[1:2], mapping = aes(x = x, y = y, colour = KMeansData_Group1$cluster)) +
  geom_point(mapping = aes_string( x = rk$centers[, "x"], y = rk$centers[, "y"]), color = "red", size = 4)

                      