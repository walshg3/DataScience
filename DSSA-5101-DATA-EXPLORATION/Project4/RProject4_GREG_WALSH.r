# R Mini-Project 4
# NAME  - Greg Walsh

# Remember to set your working directory to wherever you saved the CSV files!
setwd('/Users/gregwalsh/Github/DataScience /DSSA-5101-DATA-EXPLORATION/Project4')

# Load the required libraries
library(tidyverse) # To read the csv and manipulate strings
library(lubridate) # To manipulate dates


# Make a few decision about your final dataset.
# Document your decisions in the code.
# 1. Decide what features (variables) you want in your dataset (e.g. FirstName, LastName, DoB, DoD, Sex)
#   Do you want to have MiddleName or not?
## I went with FirstName, LastName, DOB, DOD, Sex
## No Middle Name was Used and attempts were made to remove all middle names.
# 2. Do you want to represent Sex as 'male'/'female' or 'm'/'f'?
## Represented as male and female
# 3. Decide what you will do with missing values.
#   Do you want to keep observations with no DOB? No DOD? Obviously wrong dates?
## Remove data that is incomplete other than sex. My goal was to see when a majority of the people in the dataset were born
#   If you keep them, how will you handle them?
## Did not keep them. 
#   If you delete them, are you introducing a bias?
## It might cause bias to some degree however the same argument could be made if you kept them. Depending on how much data is removed
## and what data is removed. For example if I removed people without a Sex it would remove a good chunk of the dataset causing bias
## as for Date of Birth, keeping incomplete or Obviously wrong dates would cause more bias then removing them all together. There were 
## not that many removed so the data is conclusive however there is always going to be some bias. 
# Note: Document all of your code so that a reader knows what you are doing on each line.

## Create a titles variable that will be used to clean the data of prefixes
titles <- c("mr", "mrs", "iv", "md", "phd", "iii", "ii", "and", "&", "miss", "jr", "sr", "iv", "prof", "professor", "esquire", "dr", "esq", "sc", "d", "", "III", "Jr.", "Sr.", "JR.", "SR.", "Jr", "Sr" ) 

# Read the first csv file named halloween2019a.csv
h1 <- read_csv('halloween2019a.csv')
# The first dataset has a MiddleName but no column for Sex

# ** Decide if you want to keep MiddleName or not.
# If you are not keeping MiddleName, delete the column
# ** Convert DOB and DOD to date fields
# ** Add a column for Sex with every observation set to NA

# Remove middle name Column
h1 = h1[-c(2)]
# Parse date columns correctly
h1$DOB = parse_date_time(h1$DOB,c('%m/%d/%Y',"%m%d%y","y"))
h1$DOD = parse_date_time(h1$DOD,c('%m/%d/%Y',"%m%d%y","y"))

# Remove rows with NA in them
h1 = h1[complete.cases(h1), ]

# Create Sex Column and fill with NA
h1["Sex"] = NA

# Split names by space and clean them of prefixes 
FirstName.split <- strsplit(h1$FirstName, "\\s|\\.|\\,|\\(|\\)") #Split by empty spaces, dots, commas and parenthesis
h1$FirstName <- sapply(FirstName.split, function(st) paste(st[!(st %in% titles)], collapse=" "), USE.NAMES=FALSE)
LastName.split <- strsplit(h1$LastName, "\\s|\\.|\\,|\\(|\\)") #Split by empty spaces, dots, commas and parenthesis
h1$LastName <- sapply(LastName.split, function(st) paste(st[!(st %in% titles)], collapse=" "), USE.NAMES=FALSE)


# Read the first csv file named halloween2019a.csv
h2 <- read_csv('halloween2019b.csv')
# The second dataset has no MiddleName

# ** If you decided to keep MiddleName, create a column in h2 for MiddleName, 
#      although each observation in h2 will be NA,
# If you are not keeping MiddleName, you do not have to do anything for the name here.
# ** Convert the observations of Sex to your format (i.e. male/female or m/f)
h2 = 
h2 %>%
  mutate(
    Sex = ifelse(Sex %in% c("m", "Male", "M"), "male", "female")
  )
FirstName.split <- strsplit(h2$FirstName, "\\s|\\.|\\,|\\(|\\)") #Split by empty spaces, dots, commas and parenthesis
h2$FirstName <- sapply(FirstName.split, function(st) paste(st[!(st %in% titles)], collapse=" "), USE.NAMES=FALSE)
LastName.split <- strsplit(h2$LastName, "\\s|\\.|\\,|\\(|\\)") #Split by empty spaces, dots, commas and parenthesis
h2$LastName <- sapply(LastName.split, function(st) paste(st[!(st %in% titles)], collapse=" "), USE.NAMES=FALSE)

# Parse date columns correctly
h2$DOB = parse_date_time(h2$DOB,c('%m/%d/%Y',"%m%d%y","y"))
h2$DOD = parse_date_time(h2$DOD,c('%m/%d/%Y',"%m%d%y","y"))

# Remove incomplete lines
h2 = h2[complete.cases(h2), ]

# Read the third csv file named halloween2019c.csv
h3 <- read_csv('halloween2019c.csv')
# The third dataset has Name as one field instead of three (or two)
# Some of the dates may cause problems

# ** Separate Name into FirstName, MiddleName, and LastName 
#     OR FirstName and LastName (depending on your use of MiddleName)
# ** Correct the DOB and DOD to appropriate dates
# ** If necessary, convert the observations of Sex to your format (i.e. male/female or m/f)
test.split <- strsplit(h3$Name, "\\s|\\.|\\,|\\(|\\)") #Split by empty spaces, dots, commas and parenthesis
test.clear <- sapply(test.split, function(st) paste(st[!(st %in% titles)], collapse=" "), USE.NAMES=FALSE)
h3$FirstName = sapply(strsplit(test.clear, " "), `[`, 1)
h3$LastName = str_extract(test.clear, '[^ ]+$')

# Fix Sex
h3 = h3[-c(1)]
h3 = 
  h3 %>%
  mutate(
    Sex = ifelse(Sex %in% c("m", "Male", "M"), "male", "female")
  )

# Parse date columns correctly
h3$DOB = parse_date_time(h3$DOB,c('%m/%d/%Y',"%m%d%y","y"))
h3$DOD = parse_date_time(h3$DOD,c('%m/%d/%Y',"%m%d%y","y"))

# Remove incomplete cases
h3 = h3[complete.cases(h3), ]

# ** Combine your three corrected datasets into one master dataset
# Your three corrected datasets SHOULD have the exact same column names
# You might have to rename columns in h1, h2, or h3 
#  e.g, "FirstName", "MiddleName" (optional), "LastName", "DOB", "DOD", "Sex" 
# If one dataset has First_Name and another one has FirstName, R will assume they are different!
# ** Replace h1, h2, h3 with the names you use for your corrected datasets
lifespan <- bind_rows(h1, h2, h3)

#Remove the few dates in the future. Some weird graveyard lol
lifespan = lifespan[! lifespan[["DOB"]] >= "2019-11-21", ]

# Remove duplicates in data 
lifespan = distinct(lifespan)

## 611 Total Observations

# the dataset lifespan should have 640 observations and 
# either 5 or 6 variables, depending on your use of MiddleName

## Did not have 640.. curious what you kept? 

# ** Do some analysis on lifespan 
#  (e.g. some appropriate statistics or a nice graph)
## Question: When were a majority of the people born the in graveyard and what is the standard deviation of the ages. 
## Get mean and standard dev
m<-mean(lifespan$DOB);std<-sqrt(var(lifespan$DOB))
## Creat Histogram 

hist(lifespan$DOB,
     main="Date of Birth of Graveyard",
     xlab="Year",
     ylab="Number of People",
     col="darkmagenta",
     freq=TRUE,
     breaks = 50
)
## Add lines of each datapoints
rug(lifespan$DOB,col="red")
## Create std dev curve.
lines(density(m), col="blue", lwd=2)
curve(dnorm(x, mean=m, sd=std), col="darkblue", lwd=2, add=TRUE)
