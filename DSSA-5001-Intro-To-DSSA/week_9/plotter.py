# For this assignment we will use data on abalone (a kind of sea snail). You can obtain from my DO server. It is called abalone.csv

# You can also grab a Python code there called plotter.py which is in a zip file called plotter.zip to help you get started. You will need to add some code to read in data from the abalone.csv file.

# The features in this dataset are:

# Sex / nominal / -- / M, F, and I (infant)
# Length / continuous / mm / Longest shell measurement
# Diameter / continuous / mm / perpendicular to length
# Height / continuous / mm / with meat in shell
# Whole weight / continuous / grams / whole abalone
# Shucked weight / continuous / grams / weight of meat
# Viscera weight / continuous / grams / gut weight (after bleeding)
# Shell weight / continuous / grams / after being dried
# Rings / integer / -- / +1.5 gives the age in years

# [Note that all continuous data are normalised by dividing by 200. To get the original calues multiply by 200]

# You will create a scatter plot that shows 4 features of the data.

# Create an x-y plot of length versus diameter using using Tufte's principles. Create a script in R or Python that runs at the command line and produces an output graphic such as PNG or JPG.

# Now add the features whole weight and shell weight using color and size to indicate these third and fourth dimensions. You will have now created a graphic with four features in a two dimensional graph.

# As always add a title, x and y labels, a legend and key and a caption that summarizes the graph. Transparency (alpha) can help. 

# Not required, but as a bonus, can you add the sex (and infancy) by using different marker types? You will have now created a graphic with 5 features in a two dimensional graph.

import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1,1)

data = pd.read_csv("abalone.csv", delimiter = ',', names = ['Sex', 'Length',  'Diameter', 'Height', 'Whole-weight', 'Shucked-weight', 'Viscera-weight', 'Shell-weight', 'Rings'  ])
cmap = plt.cm.summer


feature1 = data[['Length']]
feature2 = data[['Diameter']]
feature3 = data[['Whole-weight']]
feature4 = data[['Shell-weight']].values

plot1 = ax.scatter(feature1,feature2, s=200*feature3, c=feature4, cmap=cmap)
txt = "Abalone Data showing Length vs Diameter, Whole Weight is depicted in Circle Length and Shell weight is depicted in Color"
plt.figtext(0.5, 0.007, txt, wrap=False,horizontalalignment='center', fontsize=6)

plt.colorbar(plot1).set_label("Shell Weight")


ax.set_xlabel('Length')
ax.set_ylabel('Diameter')
ax.set_title('Abalone Data')


plt.savefig('figure.png')
	
