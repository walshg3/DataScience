# https://www.kaggle.com/mirichoi0218/insurance#insurance.csv
import argparse

# Setup arguments from command line : and the default options/configuration of the program
# : Filename is a required argument for the program to run
# : The remaining show/hide flags can be used to alter the default behavior of the printed results
parser = argparse.ArgumentParser(description="Program for viewing unique values in dataset")
parser.add_argument('--filename',      dest='filename',       type=str, help='The input file to process', required=True)

parser.add_argument('--show_age',      dest='print_age',      action='store_true')
parser.add_argument('--hide_age',      dest='print_age',      action='store_false')
parser.add_argument('--show_sex',      dest='print_sex',      action='store_true')
parser.add_argument('--hide_sex',      dest='print_sex',      action='store_false')
parser.add_argument('--show_bmi',      dest='print_bmi',      action='store_true')
parser.add_argument('--hide_bmi',      dest='print_bmi',      action='store_false')
parser.add_argument('--show_children', dest='print_children', action='store_true')
parser.add_argument('--hide_children', dest='print_children', action='store_false')
parser.add_argument('--show_smoker',   dest='print_smoker',   action='store_true')
parser.add_argument('--hide_smoker',   dest='print_smoker',   action='store_false')
parser.add_argument('--show_region',   dest='print_region',   action='store_true')
parser.add_argument('--hide_region',   dest='print_region',   action='store_false')
parser.add_argument('--show_charges',  dest='print_charges',  action='store_true')
parser.add_argument('--hide_charges',  dest='print_charges',  action='store_false')
parser.set_defaults(
  print_age      = True,
  print_sex      = True,
  print_bmi      = False,
  print_children = True,
  print_smoker   = True,
  print_region   = True,
  print_charges  = False,
)
args=parser.parse_args()

# Initialize sets for each feature
set_age      = set()
set_sex      = set()
set_bmi      = set()
set_children = set()
set_smoker   = set()
set_region   = set()
set_charges  = set()

# Load data into a list of strings
print("Reading: %s" % args.filename)
with open(args.filename, "r") as f:
  list_data = f.readlines()

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
if args.print_age:
  print("\n%d Unique Age Values:" % len(set_age))
  print(sorted(set_age))
if args.print_sex:
  print("\n%d Unique Sex Values:" % len(set_sex))
  print(sorted(set_sex))
if args.print_bmi:
  print("\n%d Unique BMI Values:" % len(set_bmi))
  print(sorted(set_bmi))
if args.print_children:
  print("\n%d Unique Children Values:" % len(set_children))
  print(sorted(set_children))
if args.print_smoker:
  print("\n%d Unique Smoker Values:" % len(set_smoker))
  print(sorted(set_smoker))
if args.print_region:
  print("\n%d Unique Region Values:" % len(set_region))
  print(sorted(set_region))
if args.print_charges:
  print("\nUnique Charges:")
  print(sorted(set_charges))
