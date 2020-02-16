## Greg Walsh
## Machine Learning Fundamentals 
## KMeans Implementation
## Spring 2020 
## Set Working Directory
setwd("/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/KMeans")
## Imports Libraries
library(tidyverse)
## This library needed to be installed from CRAN. It is used in the grid plots
## Run install.packages("gridExtra") in the RStudio Console to download. 
library(gridExtra)
## K Means Clustering

### Algorithm
# Step 1: Choose groups in the feature plan randomly
# Step 2: Minimize the distance between the cluster center and the different observations (centroid). It results in groups with observations
# Step 3: Shift the initial centroid to the mean of the coordinates within a group.
# Step 4: Minimize the distance according to the new centroids. New boundaries are created. Thus, observations will move from one group to another
# Repeat until no observation changes groups
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
KMeansData_Group1 <-
  as.matrix(read.csv("KMeansData_Group1.csv", header = FALSE))
colnames(KMeansData_Group1) <- c("x", "y")
kmeansnew <- function(data, K, stopval = 10e-5) {
  ## Init Variables
  # 1. Create k points for starting centroids
  # Centroids
  centroids <- data[sample.int(nrow(data), K), ]
  # Cluster 
  # rep will replicate the data https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/rep
  # Creating an exact matrix as the data but with 0's to track what cluster each data point belongs to
  cluster <- rep(0, nrow(data))
  # When to stop
  curstopval <- Inf
  ## Variables to help determine convergence
  convergence <- FALSE
  # Counter variable
  iter <- 1
  # Determine if centroids have stopped moving
  while (curstopval >= stopval && isFALSE(convergence)) {
    # +1 to counter variable
    iter <- iter + 1
    # First we want to check if the current stop value is more than or equal to the stop value
    sapply(curstopval, function(x) {
      if (x <= stopval) {
        isTRUE(convergence)
      }
    })
    # isTRUE(converged)
    prevcentroids <- centroids
    ## a. For every point in our dataset:
    for (i in 1:nrow(data)) {
      # Set Value to infinity to ensure the closest euclidian distance overwrites the value
      mindis <- Inf
      ## For every centroid:
      for (centroid in 1:nrow(centroids)) {
        ## Calculate the distance between the centroid and point
        # calculate the euclidian distance
        euclidiandist <- sum((centroids[centroid, ] - data[i, ])^2)
        # determine closest centroid to the minimum distance
        if (euclidiandist <= mindis) {
          #Assign the point to the cluster with the lowest distance
          cluster[i] <- centroid
          mindis <- euclidiandist
        }
      }
    }
    # Update Centroids
    # b. For every cluster calculate the mean of the points in that cluster
    # for each centroid
    for (i in 1:nrow(centroids)) {
      # determine new centroid based on means of the clusters
      # Assign the centroid to the mean
      centroids[i, ] <- apply(data[cluster == i, , drop = F], 2, mean)
    }
    # Recaluclate stop value to See if the centroids stopped moving
    curstopval <- vapply((prevcentroids - centroids)^2, mean, FUN.VALUE = numeric(1))
  }
  ## Used for output of function
  # save centroids as x, y centroid #
  centroids <- data.frame(centroids, cluster = 1:K)
  return(list(data = data.frame(data, cluster), centroids = centroids))
}

## Compute sum of squares within
computesumsqr <- function(kmdata, K) {
  # assign variables 
  wss <- 0.0
  # for each K value
  for (i in 1:K) {
    # for each row in the data 
    for (j in 1:nrow(kmdata$data)) {
      # if associated data point belongs to the cluster
      if (kmdata$data$cluster[j] == kmdata$centroids$cluster[i]) {
        # Computer distance from cluster to data point and add it to sum
        wss <- wss + (kmdata$data$x[j] - kmdata$centroids$x[i])^2 + (kmdata$data$y[j] - kmdata$centroids$y[i])^2
      }
    }
  }
  return(wss)
  # Removing i,j is helpful for memory management
  rm(i, j)
}

## function to send data back to user
## NOTE: plotkm will plot a grid of every cluster graph from 2:K the assignment did not call for this so it is FALSE by default
## plotelbow will graph the elbow curve like assigned
returnkmeans <- function(maxk, data, plotkm = F, plotelbow = T) {
  # Assign Variables
  # We want to run 2 thru maxK given 
  ks <- 2:maxk
  # Create a vector for the sum of squares for super fast calculations 
  wss <- vector("numeric", length(ks))
  plot_list = list()
  P <- vector("list",  length(ks))
  # for each K in ks 
  for (i in ks) {
    print(paste0("K is: ", i))
    # Run kmeansnew alg
    res <- kmeansnew(data, i)
    # If plotkm create a nice plot of every K Centroids
    if (plotkm == T) {
      data_plot1 <- rbind(res$centroids, res$data)
      p <- ggplot(data_plot1,aes(x = x, y = y, color = as.factor(cluster))) +
        geom_point() + 
        geom_point(res$centroids, mapping = aes(x = x, y = y, colour="#CC0000", size = 3 )) + 
        # Removing the legends here helps make the graph look clean
        theme(legend.position = "none", axis.title = element_blank())
      # add each plot to the P Vector 
      P[[ i ]] <- p 
    }
    #Calculate the Sum of squares within 
    wss[i - 1] <- computesumsqr(res, i)
    print(wss[i - 1])
  }
  # whoever made grobs and grids needs to be killed. This was horrible to figure out. Thank god for Stack Overflow 
  if (plotkm == T) {
    # Remove first value in list because its nothing. 
    P[[1]] <- NULL
    title <- paste("Kmeans for K = 2 -", maxk)
    grid.arrange(grobs = P, top = title , ncol=3)
  }
  # create dataframe to combine out ks and wss for easier plotting
  df <- data_frame(ks, wss)
  # If elbow plot is true, do the plot
  if (plotelbow == T) {
    print(ggplot(df,aes(x = ks,y = wss)) +
      geom_line() +
      geom_point() +
      labs(title = "The Elbow Curve of Clusters", x = "Number of Clusters (K)", y = "Within groups sum of squares"))
  }
}
# All that work to allow us to type this cool one liner. 
# Note if you want to see the clusters set plotkm = T. This will print 2 plots. 1 is the data plots per K the other is the elbow curve. Otherwise it will only plot the elbow curve.
returnkmeans(15, KMeansData_Group1, plotkm = T )
#, plotkm = TRUE
# You may get this error:
# Warning message:
#  `data_frame()` is deprecated, use `tibble()`.
# You can ignore it. Its part of the gridExtra stuff. 
# Or Error in if (x <= stopval) { : missing value where TRUE/FALSE needed
# Just rerun the code. I really dont know how to fix this.

#, plotkm = TRUE
## Test with actual kmeans
# KMeansData_Group1 <- as.matrix(read.csv("KMeansData_Group1.csv", header = FALSE))
# rk <- kmeans(KMeansData_Group1, centers = 5)
# KMeansData_Group1$cluster <- as.character(rk$cluster)
# ggplot() + geom_point(data = KMeansData_Group1[1:2], mapping = aes(x = x, y = y, colour = KMeansData_Group1$cluster)) +
#  geom_point(mapping = aes_string( x = rk$centers[, "x"], y = rk$centers[, "y"]), color = "red", size = 4)
