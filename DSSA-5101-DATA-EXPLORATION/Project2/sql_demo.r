#**SQL Demo**
# Dr. Clif Baldwin, October 2019
# A demonstration of using SQL with data frames for _Data Exploration_, week 5.

#setwd("/media/clif/3763-3663/Stockton/Data Exploration")

library(tidyverse)
library(sqldf)

load(file = "accident.RData")
load(file = "location.RData")

# How man records (or observations) are in the accident data frame?
# SQL 
sqldf("select count(*) from accident;")
# Base R
nrow(accident)

# How many observations are animal-related (i.e. CrashTypeCode = '12')?
# SQL
sqldf("select COUNT(*) from accident where CrashTypeCode = '12';")
# Base R (wrong!)
nrow(accident[accident$CrashTypeCode == "12",])
length(which(is.na(accident$CrashTypeCode)))
# Corrected Base R
nrow(accident[!is.na(accident$CrashTypeCode) & accident$CrashTypeCode == "12",])

# List the months of animal-related accidents
# SQL 
sqldf("select strftime('%m', CrashDate) AS Month from accident where CrashTypeCode = '12';")
# Base R
accident$Month <- as.factor(format(accident$CrashDate, "%m"))
accident[!is.na(accident$CrashTypeCode) & accident$CrashTypeCode == "12",]$Month

# What RoadSystems are in the location data frame?
# SQL 
sqldf("select DISTINCT RoadSystem from location")
# Base R
unique(location$RoadSystem)

# What Routes are identified as RoadSystem = '03'?
# SQL
sqldf("select Route from location where RoadSystem = '03'; ")
# Base R
subset(location, RoadSystem == '03')$Route

# What unique Routes are identified as RoadSystem = '03'?
# SQL
sqldf("select DISTINCT Route from location where RoadSystem = '03'; ")
# Base R
unique(subset(location, RoadSystem == '03')$Route)

# What unique Routes and CrashLocations are identified as RoadSystem = '03'?
# SQL
sqldf("select DISTINCT Route, CrashLocation from location where RoadSystem = '03'; ")
# Base R
unique(subset(location, RoadSystem == '03')[,c("Route", "CrashLocation")])
# Tidyverse
location %>% filter(RoadSystem == '03') %>% select(Route, CrashLocation) %>% unique

# How many total people were injured from animal-related accidents?
# SQL
sqldf("select SUM(TotalInjured) from accident where CrashTypeCode = '12'; ")
# Tidyverse
accident %>% filter(CrashTypeCode == '12') %>%
  select(TotalInjured) %>%
  summarise(SUM = sum(TotalInjured, na.rm = TRUE))

# List the CrashDate, TotalInjured, CountyName, and Route 
#  for RoadSystem = '03' AND CrashTypeCode = '12'
# Note: CrashDate and TotalInjured are in the accident data frame
# Note: CountyName and Route are in the location data frame
# SQL
sqldf("SELECT CrashDate, TotalInjured, CountyName, Route 
      FROM accident 
      JOIN location ON accident.id = location.id 
      WHERE RoadSystem = '03' AND CrashTypeCode = '12' ;")

# Base R
m1 <- merge(accident, location, by.x="id", by.y = "id")
subset(m1, RoadSystem == '03' & CrashTypeCode == '12')[,c(2,3,7,8)]
rm(m1)

# Tidyverse
inner_join(accident, location, by= c("id" = "id")) %>% filter(RoadSystem == '03', CrashTypeCode == '12') 
