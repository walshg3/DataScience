""" 
Submit a Python code which supports a function that does the following:

Accepts a filename as a variable input
Applies an age filter to the dataset in the form of a minimum and maximum inclusive age
Applies a region filter to the dataset in the form of a comma separated string
Returns two values: the number of entries in the filtered subset and the average charge from the subset
"""
def compute_avg_charges(filename, min_age, max_age, regions):
  import pandas as pd
  #create new DataFrame
  df = pd.read_csv(filename)
  #Applies an age filter to the dataset in the form of a minimum and maximum inclusive age
  minFilter = df.age >= min_age
  maxFilter = df.age <= max_age
  #Applies a region filter to the dataset in the form of a comma separated string
  regionFilter = df.region.isin(regions.split(","))
  #Combine all Filters into one final filter to return 
  finalFilter = df[minFilter & maxFilter & regionFilter]
  return (len(finalFilter), finalFilter.charges.mean())