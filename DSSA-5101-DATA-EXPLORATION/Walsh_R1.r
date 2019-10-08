#GREG WALSH
#DSSA DATA EXPLORATION
#R MINI-PROJECT 1
#Using the fish.csv dataset, write a script of R code to read the dataset and describe it. 
#At a minimum, use the functions glimpse(), str(), summary(), dim(), nrow(), str(), head().


#Imports
library(readr)
library(dplyr)

#Read File
fish <- read_csv("fish.csv", col_names = TRUE, progress = TRUE) 

glimpse(fish)
#Shows Observations (lines) Variables (columns) and some data in each column
# Observations: 5,580
# Variables: 8
# $ Year            <dbl> 2012, 2011, 2012, 2015, 2013, 2014, 2016, 2018, 2018,…
# $ County          <chr> "Franklin", "Herkimer", "Ulster", "Oneida", "Essex", …
# $ Waterbody       <chr> "Little Trout River", "Md Settlement Lake", "Stony Ki…
# $ Town            <chr> "Burke, Constable", "Webb", "Rochester", "Forestport"…
# $ Month           <chr> "April", "October", "May", "April", "May", "September…
# $ Number          <dbl> 2480, 600, 310, 860, 290, 350, 25390, 600, 350, 2130,…
# $ Species         <chr> "Brown Trout", "Brook Trout", "Brook Trout", "Brown T…
# $ `Size (Inches)` <dbl> 8.1, 4.4, 8.6, 7.8, 8.5, 3.7, 5.1, 13.0, 8.2, 8.7, 8.…


str(fish)
#Similar to Glimpse just looks structured different

summary(fish)
#Shows some info on specific columns like:
#  Year         County           Waterbody             Town          
#  Min.   :2011   Length:5580        Length:5580        Length:5580       
#  1st Qu.:2013   Class :character   Class :character   Class :character  
#  Median :2015   Mode  :character   Mode  :character   Mode  :character  
#  Mean   :2015                                                           
#  3rd Qu.:2017                                                           
#  Max.   :2018 
dim(fish)
#Retrieves or sets dimension of an object
#[1] 5580    8
nrow(fish)
#return the number of rows or columns present in fish
#[1] 5580
head(fish)
#return last few entries in the dataset
# A tibble: 6 x 8
#    Year County  Waterbody     Town       Month   Number Species  `Size (Inches)`
#   <dbl> <chr>   <chr>         <chr>      <chr>    <dbl> <chr>              <dbl>
# 1  2012 Frankl… Little Trout… Burke, Co… April     2480 Brown T…             8.1
# 2  2011 Herkim… Md Settlemen… Webb       October    600 Brook T…             4.4
# 3  2012 Ulster  Stony Kill    Rochester  May        310 Brook T…             8.6
# 4  2015 Oneida  Black River   Forestport April      860 Brown T…             7.8
# 5  2013 Essex   Clear Pond    Ticondero… May        290 Brown T…             8.5
# 6  2014 Hamilt… Rock Pond     Indian La… Septem…    350 Brook T…             3.7

as.matrix(fish[1:10,5:7])
#Create Matrix
#      Month       Number  Species        
#  [1,] "April"     " 2480" "Brown Trout"  
#  [2,] "October"   "  600" "Brook Trout"  
#  [3,] "May"       "  310" "Brook Trout"  
#  [4,] "April"     "  860" "Brown Trout"  
#  [5,] "May"       "  290" "Brown Trout"  
#  [6,] "September" "  350" "Brook Trout"  
#  [7,] "November"  "25390" "Coho"         
#  [8,] "May"       "  600" "Rainbow Trout"
#  [9,] "May"       "  350" "Brown Trout"  
# [10,] "May"       " 2130" "Splake"