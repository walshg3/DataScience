# R Assignment 2 
# Greg Walsh
# How many records are in master?
# Get a unique list of School Leader Names
# Count the unique School Leader Names
# Get all records in master for School Year 2016-2017
# Get all Total Enrolled in enroll for Grade “All Grades”
# Get the sum of total students suspended for 2016-2017 from suspend
# Get the School Leader Name, Total Enrolled, and Total Students Suspended for each school with “All Grades” and School Year 2016-2017

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
# Get the sum of total students suspended for 2016-2017 from suspend
# Sum is 13957
sqldf("select SUM(total_students_suspended) from suspend where school_year = '2016-2017'")
# Get the School Leader Name, Total Enrolled, and Total Students Suspended for each school with “All Grades” and School Year 2016-2017 
sqldf("
        select School_Leader_Name, Total_Enrolled, total_students_suspended from master 
        join enroll on master.School_ID = enroll.School_ID
        join suspend on enroll.School_ID = suspend.school_id
        where master.School_Year = '2016-2017' and enroll.Grade = 'All Grades'
        " )


