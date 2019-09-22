# https://www.kaggle.com/mirichoi0218/insurance#insurance.csv
# [Review] Importing Python Modules
import argparse

# [Review] Python Dictionaries
file_format_info = {
  "age":      { "print_by_default":True,  "field_index":0 },
  "sex":      { "print_by_default":True,  "field_index":1 },
  "bmi":      { "print_by_default":False, "field_index":2 },
  "children": { "print_by_default":True,  "field_index":3 },
  "smoker":   { "print_by_default":True,  "field_index":4 },
  "region":   { "print_by_default":True,  "field_index":5 },
  "charges":  { "print_by_default":False, "field_index":6 }
}

# Load arguments from command line and set default argument values
parser = argparse.ArgumentParser(description="Program for viewing unique values in dataset")
parser.add_argument('--filename',      dest='filename',       type=str, help='The input file to process', required=True)

# [Review] Looping over Python Dictionaries
for field_name in file_format_info:
  parser.add_argument('--show_%s' % field_name, dest='print_%s' % field_name, action='store_true')  # [Review] Python String Formatting with the % operator
  parser.add_argument('--hide_%s' % field_name, dest='print_%s' % field_name, action='store_false')

# [Review]: Variable unpacking into Functions https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters
# [Review]: Dictionary comprehensions
parser.set_defaults(**{ "print_%s" % key: file_format_info[key]["print_by_default"] for key in file_format_info })
args=vars(parser.parse_args())

# Initialize sets for storing unique values from each feature
# [Review]: Dictionary comprehensions
sets = { key: set() for key in file_format_info }

# Load data into a list of strings
print("Reading: %s" % args["filename"])
with open(args["filename"], "r") as f:
  list_data = f.readlines()

# Iterate from the 2nd line in the file to the end
for line in list_data[1:]:
  # Extract lines into individual variables; treating data as string-types
  tokens = line.split(",")
  for key in file_format_info:
    sets[key].add(tokens[file_format_info[key]["field_index"]].strip())

# Print Results
for key in sorted(file_format_info):
  if args["print_%s" % key]:
    print("\n%d Unique %s Values:" % (len(sets["age"]), key))
    print(sorted(sets[key]))
