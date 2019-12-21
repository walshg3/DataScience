#! /usr/bin/python3
import sys
import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1,1)
x=''.join(sys.stdin.readlines()[:])
str_list=x.split('\n')
x_list = list(filter(None, str_list))
x_floats=[float(i) for i in x_list]
print('Max value = ',max(x_floats))
print('Min value = ',min(x_floats))
print('Mean value = ',sum(x_floats)/float(len(x_floats)))
plt.hist(x_floats)
plt.savefig('histogram_python.png',bbox_inches='tight')
