#! /usr/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

data <- read.csv(file=args[1],header=FALSE,sep=",")

res <- cor(data)

round(res,2)

symnum(res,cutpoints=c(-1.0,-0.5,0.0,0.5,1.0),
             symbols=c("-","<",">","+")        )
