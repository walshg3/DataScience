# R Script named asteroids2.r
# Written by Dr. Clif Baldwin
# September 2019
# My working directory is set by the following line:
#setwd("/media/clif/3763-3663/Stockton/Data Exploration")

# The R script is a text file named `asteroids2.r` that I used to save the R commands I need to explore the asteroids dataset named asteroids_part.csv
# I named the script asteroids2.r because the script I gave you last week was called asteroids1.r and this script is an enhancement of that script. Using your favorite text editor (notepad++, Geany, Atom, gedit, nano - but NOT Word or Wordpad!), you can open the file asteroids2.r and edit it. You can also edit it from within RStudio as a R Script. 
# A R script is a way to save a series of R commands, including any comments, preceded by # 
# Now let us explore the asteroids dataset.

# First we need to load the Tidyverse package, in this case we can just load readr for read_csv() and dply for glimpse()
library(readr)
library(dplyr)

# Load the dataset using read_csv (instead of read.csv)
asteroids <- read_csv("asteroids_part.csv",col_names = TRUE, progress = TRUE)

glimpse(asteroids)
# asteroids is a data frame, or more appropriately a Tibble, which is a special type of data frame
# We can access rows (observations) or columns (variables) from the data frame as needed
asteroids[1:3,] # the first three observations
asteroids[,5:7] # the fifth through seventh variables
asteroids[1:2,5:7] # the first two observations of variables 5 through 7

# We could save the desired variables to a matrix, in this case the first 10 observations
as.matrix(asteroids[1:10,5:7])

# From the NASA website, we can determine the meanings of the column names

#* full_name = object full name/designation
#* neo = Near Earth Object flag
#* pha = Potentially Hazardous Asteroid flag 
#* epoch of osculation
#* e = eccentricity
#* a = semi-major axis (au)
#* q = perihelion distance (au)
#* i = inclination (deg)
#* om = Longitude of the Ascending Node (deg)
#* w = argument of perihelion (deg)
#* ma = mean anomaly (deg)
#* n = mean motion (deg/d)
#* per = period (days)

# Now that we know the variables, we can reference them using asteroids$name, for example asteroids$full_name returns the full names (only) from the data frame

# For fun, let us adopt the following question, which we can answer from the data:
# What is the mass of the Sun?
#   We can answer the question using Newton's Law of Gravitation

# First, we need the gravitational constant
# G = 6.67408 × 10-11 m3 kg-1 s-2
# It is a constant, as its name indicates
G = 6.67408e-11

# We can see that G is stored as a single value
G

# but R is case sensitive
g
# You should have gotten an error because we have not defined g, which is not the same as G

# According to the documentation, the dataset has distances in astronomical units (au) and has orbital period in days
# Our value of G uses meters for distance and seconds for time.

# To compute each asteroids velocity (m/sec), we need their semi-major axes and periods (in days)
# The semi-major axis is measured in astronomical units (au)
au <- asteroids$a

# Now au is a vector of all the asteroids' semi-major axes
# We can convert the vector au into meters by multiplying each element by the constant 149,597,870,700
radius <- au * 149597870700

# Now we have a new vector called radius, and we can delete the vector au so we do not get confused
rm(au)

# We can do the same sort of thing to convert per (in days) to seconds
days <- asteroids[,13]  # same as asteroids$per

# R saved it as a 1-dimensional data frame instead of a true vector, but it will work the same
# We could convert it if you are worried
days <- as.vector(days[,1])

seconds <- days * 24 * 60 * 60

#  To calculate the speed of the asteroids, we just recognize that the asteroids are traveling the circumference of a (nearly) circular orbit. 
# The circumference of a circle is 2*pi*r, where our radii are the vector radius
distance <- 2. * pi * radius

# The speed is the distance divided by the time, and we can divide one vector by another
speed <- distance / seconds

# the asteroids are traveling at "speed" meters per second in their orbits

# To compute the mass of the Sun, we can use the equation from physics M = (velocity^2 * radius)/ G
# see https://imagine.gsfc.nasa.gov/features/yba/CygX1_mass/gravity/sun_mass.html
Mass <- ((speed^2) * radius) / G

# We get many values - one value for each asteroid
length(Mass)

# Take the mean, which will give us a scalar
mean(Mass)

# Uh oh! At least one of our values must be missing. We will cover missing values later! For now, let us ignore them.
mean(Mass, na.rm = TRUE)

# Actual Mass of sun 1.989 × 10^30 kg
