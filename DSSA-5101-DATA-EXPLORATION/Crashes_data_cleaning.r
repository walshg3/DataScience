#**Data Cleaning**
# Dr. Clif Baldwin, September 2019

# A demonstration of data cleaning for _Data Exploration_, week 4.

#setwd("/media/clif/3763-3663/Stockton/Data Exploration")
#setwd("/home/clif/Documents/DataScience/Data Exploration")

# While driving home from Stockton in Autumn a few years ago, 
# I saw a sign on the GSP that stated most accidents involving deer occur in October and November. 
# I want to explore that claim.

# Raw data for New Jersey can be retrieved from the NJ Department of Transportation website
# at http://www.state.nj.us/transportation/refdata/accident/rawdata01-current.shtm. 
# Meta-data about the data is documented in PDF files 
# at http://www.state.nj.us/transportation/refdata/accident/masterfile.shtm. 

# I downloaded the 2015 Ocean County and Atlantic County Crashes (or accidents) data sets. 
# The file is zipped, and I extracted the text file. 
# There is Ocean2015Accidents.txt and Atlantic2015Accidents.txt for the two counties. 
# Although they are "txt" files, the documentation indicates they are comma-delimited 
#  with no headers. Looking at the text file with any text editor confirms it. 
# Since all I know is the data is comma-delimited, I will try to read the Ocean County data 
#  with read.csv()

# For this demonstration, I will use only the Atlantic County data.
# (everything I do here applies to both the Atlantic and Ocean Counties data)

# First determine how many lines (rows or observations) are in this data file. 
# From the Linux command line, `wc -l Atlantic2015Accidents.txt`
# clif@clif-pc:~$ wc -l Atlantic2015Accidents.txt
#  8020 Atlantic2015Accidents.txt

# Now we will read the Atlantic County file into R.
crashes1 <- read.csv("Atlantic2015Accidents.txt", header = FALSE, quote = "", fill = FALSE)
# Error!

# I could trying the Ocean County dataset 
# to determine if there is something specific to Atlantic County causing the error. 
# Also I am going to do it the Tidyverse way, because there are more "correction" features 
#  built into readr (e.g. read_csv) than base R (e.g. read.csv).
# I need to load the Tidyverse library
library(tidyverse)

# Since I might have to read it several times, I will save the filename as a variable.
fname <- "Atlantic2015Accidents.txt"
crashes2 <- read_csv(fname)

# Ok, multiple problems encountered!

# The easiest fix among the problems detected is to add column names. 
# I had to read the documentation at the NJ state website to determine appropriate column names.
colNames <- c("YearCountyMunicipalityCaseNumber","CountyName","MunicipalityName","CrashDate","CrashDayOfWeek",
"CrashTime","PoliceDepartmentCode","PoliceDepartment","PoliceStation","TotalKilled","TotalInjured",
"PedestriansKilled","PedestriansInjured","Severity","Intersection","AlcoholInvolved","HazMatInvolved",
"CrashTypeCode","TotalVehiclesInvolved","CrashLocation","LocationDirection","Route","RouteSuffix",
"StandardRouteIdentifier","MilePost","RoadSystem","RoadCharacter","RoadSurfaceType","SurfaceCondition",
"LightCondition","EnvironmentalCondition","RoadDividedBy","TemporaryTrafficControlZone","DistanceToCrossStreet",
"UnitOfMeasurement","DirectionFromCrossStreet","CrossStreetName","IsRamp","RampToFromRouteName",
"RampToFromRouteDirection","PostedSpeed","PostedSpeedCrossStreet","Latitude","Longitude","CellPhoneInUseFlag",
"OtherPropertyDamage","ReportingBadgeNo")

#Now let us read it again and see what problems remain. 
crashes2 <- read_csv(fname, col_names = colNames, quote="")

#Looks like problems remain. How many rows were read?
nrow(crashes2)

# There are 8020 rows in the raw text file (remember from our wc CLI)!
# Also the columns Latitude and Longitude were read as logical. 
# That does not make sense. I can specify how to read each column.
rm(crashes2) # First remove the data from R to start new.
crashes2 <- read_csv(fname, col_names = colNames, col_types = "ccccccccciiiicccccicciccdcccccccciccclcciiccccc", quote="")

# Problems persist, although I have cleaned up the column names and the column types. 
# It appears row 1763 is one of the remaining problems.
crashes2[1763,46:47]

# What does it think row 1764 is?
crashes2[1764,1:2]

# Now we see the big problem, although we had to clean all the little problems to get here. 
# We can look at the file using a text editor or 
#  from the command line using AWK to see if we see the problem.
# `awk  -F "," '{print $1;}' Atlantic2015Accidents.txt`
# We see some non-dates in the first field that contains a date code. 
# Apparently there is some code in the field OtherPropertyDamage 
#  that causes R to split the line before it really ends. 
# I need to look at the file with a text editor that displays hidden codes.
# Upon further inspection with a text editor, there is an additional linefeed code (i.e. "\n") 
#  in about fifty of the OtherPropertyDamage fields.

#We have several options at this point:
# 1) We can modify the CSV files by hand (but there are about 50 problems and we might miss some). 
 ## Not deirable at all!

# 2) We can ask Prof. Mick to write a Python file to read and clean the data. 
 ## Data Science is best conducted by teams.  Asking for help might be the best answer. 
 ## So this option is a good solution except we need to ask someone else to help. 
 ## If we do not want to wait for Prof. Mick, we need more options.

# 3) We can write R code to clean the data. 
 ## I tried this option, and it can be done. But the solution is very complicated! 
 ## Furthermore the solution I found was a way to get R to perform Linux command line functions.

# 4) We can ask Dr. Manson to write a Linux tool to clean the CSV files. 
 ## A good answer except Dr. Manson just responded "That is a good problem" when I asked him.

# 5) We can write our own Linux command to clean the file at the command line.

# After many Google searches and reading Linux documentation, I found three possible answers 
# (and all work)
# The easiest solution from the command line is
## `tr -d '\n' < Atlantic2015Accidents.txt | tr '\r' '\n' > Atlantic2015Accidents.csv`
# This solution strips out all the "\n" and then replaces "\r" with "\n". Works great on Ubuntu, but I am not sure if it would work as well in Windows.
 
# We can do the same thing using `sed` from the command line,
## `sed ':a;N;$!ba;s/\n//g' Atlantic2015Accidents.txt | tr '\r' '\n' > Atlantic2015Accidents.csv`

# Or if we want to get fancy, we can use AWK. I believe this solution will create a file 
#  that is fixed on both Linux and Windows. Although it took more work to code, 
#  it is more versatile than the previous two.
# Since we know the 46th field is the offending column, we can remove "\n" from that column only.
##`awk 'BEGIN{FS=OFS=","; RS="\r\n"}{gsub(/\n/," ", $46)} 1' Atlantic2015Accidents.txt > Atlantic2015Accidents.csv`

# If we do not know what field the offending linefeeds are in, we can clear them all, 
#  except for the true end-of-lines.
## `awk 'BEGIN{FS=OFS=","; RS="\r\n"}{gsub(/\n/," ")} 1' Atlantic2015Accidents.txt > Atlantic2015Accidents.csv`

# If we check the original file against our corrected file, we can see a difference.
## `wc -l Atlantic2015Accidents.txt`
# There are 8020 rows, as we have been seeing.

## `wc -l Atlantic2015Accidents.csv`
# After correcting the file, there are 7972 rows!

# Now we can ead in the corrected file with all of our other "corrections" to get a "good" data frame. I say "good" because there may be additional problems, such as missing data and incorrect entries, but it should be good enough to start working in R.
crashesDF <- read_csv("Atlantic2015Accidents.csv", col_names = colNames, col_types = "ccccccccciiiicccccicciccdcccccccciccclcciiccccc", quote="")

# 7971 records, which is correct. 
# I know `wc` showed 7972, but there is an extra blank row at the end 
# (I cheated and looked with `tail Atlantic2015Accidents.csv`).

# Remove the "bad" R data
rm(crashes2)

# Our research / data question is "are there more animal-related aaccidents in October and November?"
# So I want to explore the major highways for animal-related accidents and 
#  determine which months are the most dangerous. 

# There is a Tidyverse command called glimpse() to view the data.frame
glimpse(crashesDF)

# There are many missing values! 
# We can do many things to address the missing values, and we usually have to do something. 
# We can delete each observation (i.e. row) that contains a missing value. 
# Deleting the entire observation is known as listwise deletion. 
# It ensures only complete observations are used in case 
#  there is a problem with the remaining data in the observations with missing data.
# If we do not want to lose that many observations and 
#  we are confident that a missing value does not ruin the remaining data in the observation, 
#  we can just omit the missing values when the column (or feature) 
#  containing a missing value is used. 
# Deleting just the missing values when encountered is known as pairwise deletion. 
# It preserves more data but can mess up statistics by allowing different sample sizes 
#  depending on what columns are used. 
# In many cases we want to fill in the missing values. 
# If we can estimate the missing value, we might do that. 
# Of course we might be wrong, and we need to consider that. 
# Let us say that we observe all highways have a posted speed of 50 mph or greater. 
# Furthermore most highways (the average) have a posted speed of 55 mph. 
# Then we can estimate that any highway with a missing value 
#  for its posted speed should be 55 mph. 
# There are other statistical techniques to estimate missing values if needed. 
# Fortunately in the world of big data, there are usually enough observations 
#  for us to omit the missing values without affecting the outcome. 

# However, I want specific columns that address my goal, and 
#  that might eliminate the missing values.
names(crashesDF[c(4,11,16,17,18,20,22,25,26,29,30,31,41,45)])

# I will just save those columns.
crashesDF <- crashesDF[c(4,11,16,17,18,20,22,25,26,29,30,31,41,45)]

glimpse(crashesDF)

# The NJDOT Website http://www.state.nj.us/transportation/refdata/accident/crash_detail_summary.shtm 
#  describes some of the data fields. 
# CrashTypeCode== 12 indicates number of crashes where an Animal is involved
nrow(crashesDF[crashesDF$CrashTypeCode== "12",])

# What percentage of accidents are animal related?
nrow(crashesDF[crashesDF$CrashTypeCode== "12",]) / nrow(crashesDF)

# Only 3% of the data is animal related, which puts the data into some context, 
#  but remember the data question! 

# I do not want to have to search on text fields for the chosen highways 
#  because I do not trust that there are no misspellings and typos. 
# The documentation does not provide the codes for RoadSystem, 
#  but I want to know what code indicates major highway. 
# I can check what RoadSystem applies to the Garden State Parkway.
crashesDF[grep("GARDEN STATE", crashesDF$CrashLocation, ignore.case=T),]$RoadSystem

# If the GSP is RoadSystem "03", I think I want to know about accidents 
#  that occur on RoadSystem with the character "03" 
#  (or number 3 if we used read.csv and translated the field to an integer). 
# I can check to see what roads have RoadSystem 03 
#  (I use `unique` so that I do not have to look through all the repeated rows).
unique(crashesDF[crashesDF$RoadSystem == "03",]$CrashLocation)

# That list looks appropriate.  
# And notice that the Garden State Parkway is also called GSP. 
# I am going to want to see the entire dataset again. 
# So I will subset the data frame into a new data frame.
highway <- crashesDF[crashesDF$RoadSystem == "03",]
summary(highway)

# Notice that all features are complete except the posted speed, 
#  which has 9 missing values (NA) out of 1148 observations. 
# That is such a small percentage that I think I can safely 
#  omit the observations with missing values (listwise deletion). 
# In other words, I am not going to estimate the posted speed to fill in the blanks.

# As a check that we have the correct RoadSystem, we can look at the new data frame
unique(highway$RoadSystem)

# Let us look at the Route field
unique(highway$Route)

# I suspect route 444 and 446 may be the major highways in Atlantic County. 
# Since I will want to do further analysis at a later time, 
#  I am going to save this cleaned-up data frame.
highway <- crashesDF[crashesDF$RoadSystem== "03" & (crashesDF$Route == 446 | crashesDF$Route == 444),]
# saveRDS(highway, file="crashes.Rdata") # if I want to save the R dataset in R format

#According to the summary, some of the PostedSpeed entries are missing. 
# We can verify this with is.na().
is.na(highway$PostedSpeed)

# That is just messy! Let us try something else.
which(is.na(highway$PostedSpeed))

# Okay, so observation 72 is missing, and 220, and 221, and...
# How many observations are missing?
length(which(is.na(highway$PostedSpeed)))

# We had 9 missing values when we looked at only RoadSystem "03" 
#  but now we have 12 missing values. 
# I guess a few of the Route 444 and 446 are not listed as RoadSystem "03"?? 

# In addition to NAs, I need to address any outliers. 
# A problem is determining a true outlier from a true extreme data point. 
# We would like to eliminate outliers that are caused by noise (or other external influences) 
#  but we want to keep proper data even if it is extreme. 
# Well, that is true, but any extreme data observations may skew our analysis. 
# So we may have to choose our statistics based on the presence of extreme values. 
# To see how an outlier can influence a data study, let us look at TotalInjured. Ignore for a moment that we would not want to give a point estimate statistic alone. Also, let us ignore any observations with NA.
mean(crashesDF$TotalInjured, na.rm = TRUE)

# So it looks like someone gets hurt in about a third of all accidents. 
# Looking at the data in a boxplot is revealing.
boxplot(crashesDF$TotalInjured, na.rm = TRUE)

# The graph indicates few accidents result in multiple injuries. 
nrow(subset(crashesDF, TotalInjured > 5))

# Only 12 out of `r nrow(crashesDF)` have more than five people injured. 
# Do those few extreme cases skew our results??

# Perhaps instead of the arithmetic mean, we want the median 
#  to represent the average injuries per accident.
median(crashesDF$TotalInjured, na.rm = TRUE)

# While the mean may mislead the reader into thinking there are more injuries than actual, 
#  the median misleads the reader into thinking there are no injuries per accident. 
# In this case, we do not want to represent the data with an average at all. 
# The actual percentage of accidents with injuries 
nrow(subset(crashesDF, TotalInjured > 0)) / nrow(crashesDF)

# Another case of outliers is found with CrashTypeCode. 
# Let us display a histogram of the Crash Types involved in accidents. 
hist(as.integer(crashesDF$CrashTypeCode))

# CrashTypeCode is not truly a numeric variable since the codes represent categories. 
# But, the histogram is informative. 
# Here is a case where the value 99 is a true (extreme) data value. 
# Apparently 99 is the code for "Other". 

## Additional Information to Compliment This Week’s Topics
#* R Programming for Data Science (ch. 10, 11,12)
#* The blog post on missing data at http://uc-r.github.io/missing_values 
#* The blog post at https://www.r-bloggers.com/showing-some-respect-for-data-munging/