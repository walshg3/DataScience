def compute_avg_charges(filename, min_age, max_age, regions):
  with open(filename) as csv_file:
    charges = []
    next(csv_file)
    for row in csv_file:
      line = row.split(',')
      if int(line[0]) >= min_age and int(line[0]) <= max_age and line[5] in regions.split(","):
        charges.append(line[6].rstrip())
    charges = [float(i) for i in charges]
  return(len(charges), sum(charges)/len(charges))