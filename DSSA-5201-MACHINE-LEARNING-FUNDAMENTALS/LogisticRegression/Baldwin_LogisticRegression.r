## Dr. Clif Baldwin
## October 23, 2017; revised April 16, 2018
## Based on code from https://dernk.wordpress.com/2013/06/03/gradient-descent-for-logistic-regression-in-r/
## For Machine Learning

library(ggplot2)

sigmoid = function(z){
  1 / (1 + exp(-z))
}

#cost function
cost = function(X, y, theta){
  m = nrow(X)
  hx = sigmoid(X %*% theta)
  (1/m) * (((-t(y) %*% log(hx)) - t(1-y) %*% log(1 - hx)))
}

gradient = function(X, y, theta){
  m = nrow(X)
  X = as.matrix(X)
  hx = sigmoid(X %*% theta)
  (1/m) * (t(X) %*% (hx - y))
}

logisticRegression <- function(X, y, maxiteration) {
  alpha = 0.001 
  X = cbind(rep(1,nrow(X)),X) 
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X)) 

  for (i in 1:maxiteration){
    theta = theta - alpha * gradient(X, y, theta)
  }
  return(theta)
}

logisticPrediction <- function(betas, newData){  
  X <- na.omit(newData)
  X = cbind(rep(1,nrow(X)),X) 
  X <- as.matrix(X)
  return(sigmoid(X%*%betas))
}

testLogistic <- function(train, test, threshold) {
	glm1 = glm(Y~.,data=train,family=binomial)
	TestPrediction = predict(glm1, newdata=test, type="response")
	vals <- table(test$Y, TestPrediction > threshold)
	accuracy = (vals[1]+vals[4])/sum(vals) 
	print(paste("The R accuracy of the computed data",accuracy, sep = " "))
	sensitivity = vals[4]/(vals[2]+vals[4])
	specificity = vals[1]/(vals[1]+vals[3])
	print(paste("The R sensitivity of the computed data",sensitivity, sep = " "))
	print(paste("The R specificity of the computed data",specificity, sep = " "))
}

ROC <- function(train, test) {
  glm1 = glm(Y~.,data=train,family=binomial)
  TestPrediction = predict(glm1, newdata=test, type="response")
  
  sensitivity = vector(mode = "numeric", length = 101)
  falsepositives = vector(mode = "numeric", length = 101)
  thresholds = seq(from = 0, to = 1, by = 0.01)
  for(i in seq_along(thresholds)) {
    vals <- table(test$Y, TestPrediction > thresholds[i])
    sensitivity[i] = vals[4]/(vals[2]+vals[4])
    falsepositives[i] = vals[3]/(vals[1]+vals[3]) # false positives, or 1 - specificity
  }

  ggplot() + 
    geom_line(aes(falsepositives, sensitivity), colour="red") +
    geom_abline(slope = 1, intercept = 0, colour="blue") +
    labs(title="ROC Curve", x= "1 - Specificity (FP)", y="Sensitivity (TP)") +
    geom_text(aes(falsepositives, sensitivity), label=ifelse(((thresholds * 100) %% 10 == 0),thresholds,''),nudge_x=0,nudge_y=0)
}


### Data Preparation ###
# The provided training data
data1 <- read.csv(".csv", header = TRUE)

library(caTools) # To split the data - you can use any technique to split the data if you prefer
set.seed(88)
split = sample.split(data1$Y, SplitRatio = 0.75)
train = subset(data1, split == TRUE)
test = subset(data1, split == FALSE)
rm(split)

X <- train[,-ncol(train)]
y <- train[,ncol(train)]

### End Data Preparation ###

maxiteration = 150000
betas <- logisticRegression(X, y, maxiteration)

X <- test[,-ncol(test)]
predictedTest <- logisticPrediction(betas, X)

threshold = 0.5
vals <- table(test$Y, predictedTest > threshold)
accuracy = (vals[1]+vals[4])/sum(vals)
print(paste("The accuracy of the computed data",accuracy, sep = " "))
sensitivity = vals[4]/(vals[2]+vals[4])
specificity = vals[1]/(vals[1]+vals[3])
print(paste("The sensitivity of the computed data",sensitivity, sep = " "))
print(paste("The specificity of the computed data",specificity, sep = " "))

testLogistic(train, test, threshold)

ROC(train, test)
