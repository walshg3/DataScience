#!/bin/bash
# Greg Walsh
# Created October 28th 2019
# Last Edit: October 28th 2019
# Details: Gets a book online saves to a .txt file. Cleans the information. Then will search the Density or Actual Word count of words from parameters 
# Usage: Prompt$ ./bookscript booklink density/actual word1 word2 word3 word4 word5 ... 
# Example: ./bookscript https://www.gutenberg.org/files/84/84-0.txt density monster Frankenstein scared love hate and

## Curl a book file. 
filename=$1

curl $filename > book.txt

## Use this for linux
#sed -i '/^$/d' $1

## I am on a Mac so I needed this to do the same
sed -i "" '/^[[:space:]]*$/d' book.txt
## split the file into files of 500 lines per 
NUM_LINES=$(wc -l book.txt | awk '{print $1}')
split -l $2 book.txt
SPLITS=$(ls | grep xa) 
COUNTER=0
x='||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'


# If density is chosen run this block
if [ $3 == "density" ]
then
	echo "USING DENSITY"
	# for loop everything from 4th parameter (start of words)
	for var in "${@:4}"
	do
		echo "searching for the word: $var"
		# Echo bar graphs with params from user
		echo "|---5----10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---100-->"
		for file in $SPLITS
		do  
			# Get Word Count from each split file per word.
			WORD_COUNT=$(grep -o $var $file | wc -l)
			if (($WORD_COUNT == 0))
			then
				echo "NA" 
			else 
				# Do some math. Density can not be a decimal so we must multiply by 1000 to ensure no decimal value
				DENSITY=$(echo "$WORD_COUNT / $2" | bc -l)
				PLOT_DATA=$(echo "$DENSITY * 1000 / 1" | bc)
				if (( $PLOT_DATA >= 100 ))
				then
				echo "WARNING: Data is to high to graph actual value $PLOT_DATA"  
				else
				echo $x | cut -c1-$PLOT_DATA
				fi
			fi
			COUNTER=$((COUNTER+1))
		done
		echo "|---5----10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---100-->"
		COUNTER=0
	done 
# If actual is chosen run this block
elif [ $3 == "actual" ]
then
	echo "USING ACTUAL WORD COUNT"
	for var in "${@:4}"
	do
		echo "searching for the word: $var"
		# Echo bar graphs with params from user
		echo "|---5----10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---100-->"
		for file in $SPLITS
		do  
			WORD_COUNT=$(grep -o $var $file | wc -l)
			PLOT_DATA=$(echo "$WORD_COUNT / 1" | bc)
			if (($WORD_COUNT == 0))
			then
				echo "NA" 
			else 
				if (( $PLOT_DATA >= 100 ))
				then
				echo "WARNING: Data is to high to graph. Actual value" $PLOT_DATA  
				else
				echo $x | cut -c1-$PLOT_DATA
				fi
			fi
			COUNTER=$((COUNTER+1))
		done
		echo "|---5----10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---100-->"
		COUNTER=0
	done 
else echo "Error please chose density or actual"
fi