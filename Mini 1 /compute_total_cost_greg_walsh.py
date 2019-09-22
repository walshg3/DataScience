""" 
Submit a Python code which supports a function that does the following:

Accepts a filename as a variable input
Applies an age filter to the dataset in the form of a minimum and maximum inclusive age
Applies a region filter to the dataset in the form of a comma separated string
Returns two values: the number of entries in the filtered subset and the average charge from the subset
"""

def compute_avg_charges(filename, min_age, max_age, regions):
  import pandas as pd
  df = pd.read_csv(filename)
  sf = df[
    (df.age >= min_age)
    & 
    (df.age <= max_age)
    &
    (df.region.isin(regions.split(",")))
  ]
  return ( len(sf), sf.charges.mean() )


def compute_avg_charges_test(filename,min_age, max_age, regions):
  