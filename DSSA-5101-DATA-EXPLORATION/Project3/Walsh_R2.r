## Greg Walsh
## Assignment 2 
## Data Exploration Fall 2019 
# Create a basic plot of Yield per Colony against All Neonics
# Create a basic histogram of the number of honey producing colonies
# Create a plot using ggplot() or qplot() of Yield per Colony against All Neonics
# Create a histogram using ggplot() or qplot() of the number of honey producing colonies
# Convert year, state, and Region into Factors
# Create a histogram using ggplot() of Total production of honey by State
# Create a bar graph using ggplot() of the Number of colonies by Year
# Create a bar graph using ggplot() of the Number of colonies per State with indication of number of colonies each Year
# Create a line graph using ggplot() with each line representing a different state and indicating the Number of colonies per Year
# Given the data, present a graph that you find interesting. You can use any graphing functions you want (e.g. plot, qplot, ggplot, plot.ly). It should be easy to recognize what you are presenting (thanks to titles, labels, and whatever else it needs). 
setwd("/Users/gregwalsh/Github/DataScience /DSSA-5101-DATA-EXPLORATION/Project3")
library(tidyverse)
library(sqldf)
library(ggplot2)
## Load DataSet 
honeybee <- read.csv(file="HoneyBees.csv",sep=",",head=TRUE)
## Create a basic plot of Yield per Colony against All Neonics
plot(honeybee$yieldpercol, honeybee$nAllNeonic)
## Create a basic histogram of the number of honey producing colonies
hist(honeybee$numcol)
## Create a plot using ggplot() or qplot() of Yield per Colony against All Neonics
ggplot(honeybee, aes(x=yieldpercol,y=nAllNeonic)) + geom_point(binwidth = 10)
## Create a histogram using ggplot() or qplot() of the number of honey producing colonies
ggplot(honeybee, aes(x=numcol)) + geom_histogram(bins = 30)
## Convert year, state, and Region into Factors
year_factor <- factor(honeybee$year)
state_factor <- factor(honeybee$state)
region_factor <- factor(honeybee$Region)
## Create a histogram using ggplot() of Total production of honey by State
ggplot(honeybee, aes(x=honeybee$state, y=honeybee$totalprod)) + geom_histogram(stat = "identity")
## Create a bar graph using ggplot() of the Number of colonies by Year
ggplot(honeybee, aes(x=honeybee$year, y=honeybee$numcol)) + geom_bar(stat = "identity")
## Create a bar graph using ggplot() of the Number of colonies per State with indication of number of colonies each Year
ggplot(honeybee, aes(x=state_factor, y=honeybee$numcol, fill=year_factor)) + geom_bar(stat = "identity", position = "stack")
## Create a line graph using ggplot() with each line representing a different state and indicating the Number of colonies per Year
ggplot(honeybee, aes(x=honeybee$year, y=honeybee$numcol, group=honeybee$state)) + geom_line(aes(color=state_factor))
## Given the data, present a graph that you find interesting. You can use any graphing functions you want (e.g. plot, qplot, ggplot, plot.ly). It should be easy to recognize what you are presenting (thanks to titles, labels, and whatever else it needs). 
## Question: Which states have the most agressive change in price per pound of honey per year?
ggplot(honeybee, aes(x=honeybee$year,y=honeybee$priceperlb)) + geom_line() + facet_wrap(facets = vars(honeybee$state)) + ggtitle("Price per Pound of Honey By State Per Year") + labs(y="Price Per LB", x = "Year")
## Answer NJ ME IL 
       