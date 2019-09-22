# Load data into a list of strings
with open("insurance.csv", "r") as f:
  list_data = f.readlines()

# Initialize sets for each feature
set_age      = set()
set_sex      = set()
set_bmi      = set()
set_children = set()
set_smoker   = set()
set_region   = set()
set_charges  = set()

# Iterate from the 2nd line in the file to the end
for line in list_data[1:]:
  # Extract lines into individual variables; treating data as string-types
  age, sex, bmi, children, smoker, region, charges = line.split(",")

  # Add the individual values into the associated set
  set_age.add(age)
  set_sex.add(sex)
  set_bmi.add(bmi)
  set_children.add(children)
  set_smoker.add(smoker)
  set_region.add(region)
  set_charges.add(charges)

# Display unique values
print("\n%d Unique Age Values:" % len(set_age))
print(sorted(set_age))
print("\n%d Unique Sex Values:" % len(set_sex))
print(sorted(set_sex))
print("\n%d Unique BMI Values:" % len(set_bmi))
print(sorted(set_bmi))
print("\n%d Unique Children Values:" % len(set_children))
print(sorted(set_children))
print("\n%d Unique Smoker Values:" % len(set_smoker))
print(sorted(set_smoker))
print("\n%d Unique Region Values:" % len(set_region))
print(sorted(set_region))
print("\nUnique Charges:")
print(sorted(set_charges))
