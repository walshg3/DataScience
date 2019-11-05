
library(tidyverse)
library(sqldf)

load(file = "master.RData")
load(file = "enroll.RData")
load(file = "suspend.RData")

#glimpse(master)
#glimpse(enroll)
#glimpse(suspend)

# How many records are in master?
# 5070
sqldf("select count(*) from master;")

# Get a unique list of School Leader Names
# Total List size 1277 of unique School Leader Names
school_leader_unique = sqldf("select DISTINCT School_Leader_Name from master")
sqldf("select count(DISTINCT School_Leader_Name) from master")
# Get all records in master for School Year 2016-2017
year_16_17 = sqldf("select * from master where School_Year = '2016-2017'")
# Get all Total Enrolled in enroll for Grade “All Grades”
# Total size 215
sqldf("select count(Grade) from enroll where Grade = 'All Grades'")