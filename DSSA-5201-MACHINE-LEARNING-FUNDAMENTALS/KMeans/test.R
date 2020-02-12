## Set Working Directory 
setwd("/Users/gregwalsh/Github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/KMeans")
require(dplyr)
require(ggplot2)
library(tidyverse)
set.seed(1234)
set1=mvrnorm(n = 300, c(-4,10), matrix(c(1.5,1,1,1.5),2))
set2=mvrnorm(n = 300, c(5,7), matrix(c(1,2,2,6),2))
set3=mvrnorm(n = 300, c(-1,1), matrix(c(4,0,0,4),2))
set4=mvrnorm(n = 300, c(10,-10), matrix(c(4,0,0,4),2))
set5=mvrnorm(n = 300, c(3,-3), matrix(c(4,0,0,4),2))
DF=data.frame(rbind(set1,set2,set3,set4,set5),cluster=as.factor(c(rep(1:5,each=300))))
ggplot(DF,aes(x=X1,y=X2,color=cluster))+geom_point()




## Import Data
KMeansData_Group1 <- read_csv("KMeansData_Group1.csv")



kmeans=function(data,K=4,stop_crit=10e-5)
{
  #Initialisation of clusters
  centroids = data[sample.int(nrow(data),K),]
  #print(centroids)
  current_stop_crit = 1000
  cluster = rep(0,nrow(data))
  converged = FALSE
  it = 1
  while(current_stop_crit>=stop_crit && converged == FALSE)
  {
    it=it+1
    #ifelse(condition, do_if_true, do_if_false)
    #sapply(current_stop_crit, function(x)  if x <= stop_crit })
    print(current_stop_crit, stop_crit)
    if (current_stop_crit <= stop_crit)
    {
      converged = TRUE
    }
    old_centroids=centroids
    #print(old_centroids)
    ##Assigning each point to a centroid
    for (i in 1:nrow(data))
    {
      min_dist=10000000000
      for (centroid in 1:nrow(centroids))
      {
        distance_to_centroid=sum((centroids[centroid,]-data[i,])^2)
        #print(distance_to_centroid)
        if (distance_to_centroid<=min_dist)
        {
          cluster[i] = centroid
          min_dist = distance_to_centroid
        }
      }
    }
    ##Assigning each point to a centroid
    for (i in 1:nrow(centroids))
    {
      centroids[i,]=apply(data[cluster==i,],2,mean)
    }
    current_stop_crit = lapply((old_centroids-centroids)^2, mean, na.rm = TRUE)
  }
  return(list(data=data.frame(data,cluster),centroids=centroids))
}


res=kmeans(KMeansData_Group1[1:2],K=5)
#res=kmeans(DF[1:2],K=5)
res$centroids$cluster=1:5
res$data$isCentroid=F
res$centroids$isCentroid=T
data_plot=rbind(res$centroids,res$data)
ggplot(data_plot,aes(x=X1,y=X2,color=as.factor(cluster),size=isCentroid,alpha=isCentroid))+geom_point()
