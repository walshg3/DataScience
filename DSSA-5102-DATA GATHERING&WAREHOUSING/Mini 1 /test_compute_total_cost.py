from compute_total_cost_greg_walsh import compute_avg_charges as avc

def avcp(filename, min_age, max_age, regions):
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

def run_test(test_name, srcfile, min_age, max_age, regions, avg_tolerance = 1e-5):
  num_records_expected, avg_expected = avcp(srcfile, min_age, max_age, regions)
  num_records_computed, avg_computed =  avc(srcfile, min_age, max_age, regions)

  
  record_size_passed = num_records_computed == num_records_expected
  avg_compute_passed = abs(avg_computed - avg_expected) < avg_tolerance
  
  passed = record_size_passed and avg_compute_passed
  print("[%s]: %s" % (passed, test_name))
  if record_size_passed != True:
    print("  Record Size Mismatch")
  if avg_compute_passed != True:
    print("  Average Value Computed Mismatch")

  return passed

tests = [
  run_test("    Full Dataset", "insurance.csv",  0.0, 900.0, "northeast,southeast,northwest,southwest" ),
  run_test("  All Ages NE,NW", "insurance.csv",  0.0, 900.0, "northeast,northwest"                     ),
  run_test("  All Ages SE,SW", "insurance.csv",  0.0, 900.0, "southeast,southwest"                     ),
  run_test("  All Ages NE,SE", "insurance.csv",  0.0, 900.0, "northeast,southeast"                     ),
  run_test("  All Ages NW,SW", "insurance.csv",  0.0, 900.0, "northwest,southwest"                     ),
  run_test("     All Ages NE", "insurance.csv",  0.0, 900.0, "northeast"                               ),
  run_test("     All Ages NW", "insurance.csv",  0.0, 900.0, "northwest"                               ),
  run_test("     All Ages SE", "insurance.csv",  0.0, 900.0, "southeast"                               ),

  run_test("    0-20 Ages SW", "insurance.csv",  0.0, 20.0, "southwest"                                ),
  run_test(" 0-20 Ages NE,NW", "insurance.csv",  0.0, 20.0, "northeast,northwest"                      ),
  run_test(" 0-20 Ages SE,SW", "insurance.csv",  0.0, 20.0, "southeast,southwest"                      ),
  run_test(" 0-20 Ages NE,SE", "insurance.csv",  0.0, 20.0, "northeast,southeast"                      ),
  run_test(" 0-20 Ages NW,SW", "insurance.csv",  0.0, 20.0, "northwest,southwest"                      ),
  run_test("    0-20 Ages NE", "insurance.csv",  0.0, 20.0, "northeast"                                ),
  run_test("    0-20 Ages NW", "insurance.csv",  0.0, 20.0, "northwest"                                ),
  run_test("    0-20 Ages SE", "insurance.csv",  0.0, 20.0, "southeast"                                ),
  run_test("    0-20 Ages SW", "insurance.csv",  0.0, 20.0, "southwest"                                ),

  run_test("   21-40 Ages SW",  "insurance.csv", 21.0, 40.0, "southwest"                               ),
  run_test("21-40 Ages NE,NW", "insurance.csv",  21.0, 40.0, "northeast,northwest"                     ),
  run_test("21-40 Ages SE,SW", "insurance.csv",  21.0, 40.0, "southeast,southwest"                     ),
  run_test("21-40 Ages NE,SE", "insurance.csv",  21.0, 40.0, "northeast,southeast"                     ),
  run_test("21-40 Ages NW,SW", "insurance.csv",  21.0, 40.0, "northwest,southwest"                     ),
  run_test("   21-40 Ages NE", "insurance.csv",  21.0, 40.0, "northeast"                               ),
  run_test("   21-40 Ages NW", "insurance.csv",  21.0, 40.0, "northwest"                               ),
  run_test("   21-40 Ages SE", "insurance.csv",  21.0, 40.0, "southeast"                               ),
  run_test("   21-40 Ages SW", "insurance.csv",  21.0, 40.0, "southwest"                               ),

  run_test("   41-60 Ages SW",  "insurance.csv", 41.0, 60.0, "southwest"                               ),
  run_test("41-60 Ages NE,NW", "insurance.csv",  41.0, 60.0, "northeast,northwest"                     ),
  run_test("41-60 Ages SE,SW", "insurance.csv",  41.0, 60.0, "southeast,southwest"                     ),
  run_test("41-60 Ages NE,SE", "insurance.csv",  41.0, 60.0, "northeast,southeast"                     ),
  run_test("41-60 Ages NW,SW", "insurance.csv",  41.0, 60.0, "northwest,southwest"                     ),
  run_test("   41-60 Ages NE", "insurance.csv",  41.0, 60.0, "northeast"                               ),
  run_test("   41-60 Ages NW", "insurance.csv",  41.0, 60.0, "northwest"                               ),
  run_test("   41-60 Ages SE", "insurance.csv",  41.0, 60.0, "southeast"                               ),
  run_test("   41-60 Ages SW", "insurance.csv",  41.0, 60.0, "southwest"                               )
]

print( "Score: %.2f" % (len([x for x in tests if x ]) / len(tests) * 100 ))


