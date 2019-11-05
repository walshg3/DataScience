#!/bin/bash
# Greg Walsh
# Created October 28th 2019
# Last Edit: October 28th 2019
# Details: Gets a book online saves to a .txt file. Cleans the information. Then will search for 5 of the keywords from parameters
# Usage: Prompt$ ./bookscript booklink word1 word2 word3 word4 word5
# Example: ./bookscript https://www.gutenberg.org/files/84/84-0.txt monster Frankenstein scared love hate and
#xaa

## Curl a book file. 
filename=$1

curl $filename > book.txt

## Use this for linux
#sed -i '/^$/d' $1

## I am on a Mac so I needed this to do the same
sed -i "" '/^[[:space:]]*$/d' book.txt
## split the file into files of 500 lines per 
#wc -l book.txt 
NUM_LINES=$(wc -l book.txt | awk '{print $1}')
echo $NUM_LINES
split -l $2 book.txt
SPLITS=$(ls | grep xa) 
echo $SPLITS
## using a for loop get the density of the words per split file
COUNTER=0

for var in "${@:3}"
do
	echo "searching for the word: $var"
	# Echo bar graphs with params from user
	echo "|---5----10---15---20---25--->"
	for file in $SPLITS
	do  
		#echo "file $COUNTER"
		#echo "search var is $var"
		#echo "search Split is $file"
		WORD_COUNT=$(grep -o $var $file | wc -l)
		#echo $WORD_COUNT
		if (($WORD_COUNT == 0))
		then
			echo "NONE" 
		else 
			#PLOT_DATA=$(echo 'scale=3;'$WORD_COUNT / $2 | bc)
			echo 'scale=3;'$WORD_COUNT / $2 | bc
			#PLOT_DATA=$(echo 'scale=3;'$PLOT_DATA * 1000 | bc)
			#echo $x | cut -c1-$DENSITY
		fi
		COUNTER=$((COUNTER+1))
	done
	echo "|---5----10---15---20---25--->"
	COUNTER=0
done 

# Create Variable for the list of '|'
x='|||||||||||||||||||||||||||||'

# Echo bar graphs with params from user
#echo "|---5----10---15---20---25--->"
#echo $x | cut -c1-$1
#echo $x | cut -c1-$2
#echo $x | cut -c1-$3
#echo $x | cut -c1-$4
#echo $x | cut -c1-$5
#echo "|---5----10---15---20---25--->"
