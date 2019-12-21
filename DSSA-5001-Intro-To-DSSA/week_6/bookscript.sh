#!/bin/bash
# Greg Walsh
# Created October 28th 2019
# Last Edit: October 28th 2019
# Details: Gets a book online saves to a .txt file. Cleans the information. Then will search the Density or Actual Word count of words from parameters 
# Usage: Prompt$ ./bookscript booklink word1 word2 word3 word4 word5 ... 
# Example: ./bookscript https://www.gutenberg.org/files/84/84-0.txt  monster Frankenstein scared love hate and

## Curl a book file. 
filename=$1

curl $filename > book.txt

## Use this for linux
#sed -i '/^$/d' $1
## use this for mac
sed -i "" '/^[[:space:]]*$/d' book.txt

NUM_LINES=$(wc -l book.txt | awk '{print $1}')
split -l $2 book.txt
SPLITS=$(ls | grep xa) 
COUNTER=0
x='||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'


for var in "${@:3}"
do
	echo "searching for the word: $var"
	# Echo bar graphs with params from user
	echo "|---5----10---15---20---25---30---35---40---45---50-->"
	for file in $SPLITS
	do  
		WORD_COUNT=$(grep -o $var $file | wc -l)
		PLOT_DATA=$(echo "$WORD_COUNT / 1" | bc)
		if (($WORD_COUNT == 0))
		then
			echo "NONE" 
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
	echo "|---5----10---15---20---25---30---35---40---45---50-->"
	COUNTER=0
done 
