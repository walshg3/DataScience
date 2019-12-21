# Greg Walsh
# 10/22/19
# Script will take input of specific filename from week 4 assignments and output the assigned tasks of the assignment
# usage prompt$


if [ $1 = norway.csv ] 
then 
	echo "norway.csv"
	sed '/^$/d' $1 | 
	sed 's/Meas/Measured/g' |
	sed '/-999/d' |
	sed 's/NA,.9/13.9/g' |
	sed 's/NA,/13.9/g' |
	sed 's/,$//' |
	sed -E 's/([0-9]{1,2})\/([0-9]{1,2})\/([0-9]{4})/\1-\2-\3/'
elif [ $1 = faadata.xml ]
then
	grep -i "<gml:pos>" $1 | sed 's/<gml:pos>// ; s/<\/gml:pos>// ; s/              //g ; s/ /,/g' | head -n 20
elif [ $1 = WDB_Accused.csv ]
then
	grep -o "Male" $1 | wc -w
	grep -o "Female" $1 | wc -w
 
else 
echo "Error: File Not Found Please enter one of the following: norway.csv WDB_Accused.csv faadata.xml"
fi
