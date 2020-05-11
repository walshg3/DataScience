setwd("C:/Users/Gregwalsh96/github/DataScience/DSSA-5201-MACHINE-LEARNING-FUNDAMENTALS/DecisionTrees")


library(tidyverse)
#install.packages("rpart")
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot) # for prp()
#install.packages("randomForest")
library(randomForest)
library(caTools) #for sample.split()


## A method similar to CART is random forests.
## This method was designed to improve the prediction accuracy of CART and works by building a large number of CART trees.
## Unfortunately, this makes the method less interpretable than CART, so often you need to decide if you value the interpretability or the increase in accuracy more.


## Build a CART tree for letters_ABPR.csv - classification of letter recognition
letters = read.csv("letters_ABPR.csv")
glimpse(letters)
# Let us say I want to know if a letter is a 'B' or not
letters$isB = as.factor(letters$letter == "B")

set.seed(1000)
spl = sample.split(letters$isB, SplitRatio = 0.5)
train = subset(letters, spl==TRUE)
test = subset(letters, spl==FALSE)

CARTb = rpart(isB ~ . - letter, data=train, method="class")
prp(CARTb)
PredictTest = predict(CARTb, newdata = test, type = "class")
t1 = table(test$isB, PredictTest)
(t1[1,1]+t1[2,2])/sum(t1) # accuracy


set.seed(1000)
lettersForest = randomForest(isB ~ . - letter, data = train )
PredictForest = predict(lettersForest, newdata = test, type = "class")
t1 = table(test$isB, PredictForest)
(t1[1,1]+t1[2,2])/sum(t1) #accuracy


letters$letter = as.factor( letters$letter ) 
set.seed(2000)
spl = sample.split(letters$letter, SplitRatio = 0.5)
train = subset(letters, spl==TRUE)
test = subset(letters, spl==FALSE)
CARTc = rpart(letter ~ . - isB, data=train, method="class")
prp(CARTc)
PredictTest = predict(CARTc, newdata = test, type = "class")
t1 = table(test$letter, PredictTest)
(t1[1,1]+t1[2,2]+t1[3,3]+t1[4,4])/sum(t1) #accuracy

set.seed(1000)
lettersForest = randomForest(letter ~ . - isB, data = train )
PredictForest = predict(lettersForest, newdata = test, type = "class")
t1 = table(test$letter, PredictForest)
(t1[1,1]+t1[2,2]+t1[3,3]+t1[4,4])/sum(t1) #accuracy



## Predicting Earnings from census data
census = read.csv("census.csv")
glimpse(census)
set.seed(2000)
spl = sample.split(census$over50k, SplitRatio = 0.6)
train = subset(census, spl==TRUE)
test = subset(census, spl==FALSE)

CARTmodel = rpart(over50k ~ ., data=train, method="class") 
prp(CARTmodel)

PredictTest = predict(CARTmodel, newdata = test, type = "class")
t1 = table(test$over50k, PredictTest)
(t1[1,1]+t1[2,2])/sum(t1) #accuracy

# To show the strength of RandomForests, let us take a subset of the training data and run a RandomForest
set.seed(1)
trainSmall = train[sample(nrow(train), 2000), ]
over50Forest = randomForest(over50k ~ ., data = trainSmall )
PredictForest = predict(over50Forest, newdata = test)
t1 = table(test$over50k, PredictForest)
(t1[1,1]+t1[2,2])/sum(t1) #accuracy

