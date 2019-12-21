
indegodata=$(ls "/Users/gregwalsh/Github/DataScience /DSSA-5001-Intro-To-DSSA/week_10")

cat $indegodata > indegocombined.csv

for file in $indegodata

do 

    continue

    ### Clean the Data of " and remove the col's of every file other than one 
    #if [ ${file: -4} == ".csv" ]
    #then
    #    echo $file
    #    sed -i '' 's/\"//g' $file
    #    head -n 1 $file
    #fi

    #if [ $file == "indego-2019-q2.csv" ]
    #then
    #    echo "Found"
    #    continue
    #else
    #    sed -i '' 1d $file 
    #fi
done



