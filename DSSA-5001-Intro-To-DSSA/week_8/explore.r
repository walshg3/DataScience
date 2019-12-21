#! /usr/bin/env Rscript
# This R script reads a column of data points from the standard
# input using $ ./explore.r < filename.dat
# and calculates various statistical parameters and also creates 
# a histogram graphic.
png("boxplot.png")
d<-scan("stdin", quiet=TRUE)
cat("Minimum = ",min(d),"\n")
cat("Maximum = ",max(d),"\n")
cat("Median  = ",median(d),"\n")
cat("Mean    = ",mean(d),"\n")
summary(d)
stem(d)
boxplot(d)
