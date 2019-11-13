## Greg Walsh
## DSSA-5102-091 - DATA GATHERING & WAREHOUSING
## Fall 2019
## Mini 2
## Scrape IMDB's website for the top 250 movies and add them to an SQL Database

## Import Statements
import requests
import pandas as pd
import sqlite3
import bs4
import re 

## Get Data using requests library
response = requests.get('https://www.imdb.com/chart/top?ref_=ft_250')
## Import into Beautiful Soup's html parser
soup = bs4.BeautifulSoup(response.text, 'html.parser')

# Create data scructure 
data = {
    "title":       [],
    "year":        [],
    "imdb_rating": []
}
## tbody is the table of the movies and lister-list is the name of the class on the ste we need.
## each tr tag is a row in the table
list = soup.find("tbody", class_="lister-list").find_all("tr")
## Cycle through the rows in the list we just created and gather the movie title, Year, and IMDB Rating.
for row in list :
    data['title'].append(row.find(class_="titleColumn").find("a").text)
    years = row.find(class_="titleColumn").find(class_="secondaryInfo").text
    years = years.replace('(','')
    years = years.replace(')','')
    data['year'].append(years)
    data['imdb_rating'].append(row.find(class_="imdbRating").find("strong").text)

#Create a Pandas Dataframe with the scraped data
df = pd.DataFrame(data)

## Create the SQL Database
db     = sqlite3.connect(":memory:")
cursor = db.cursor()
cursor.execute("""
   CREATE TABLE MOVIES("TITLE" TEXT,
    "YEAR" INTEGER,
    "IMDB_RATINGS" REAL
    )
""")



## Cycle through the data in the Pandas Data Frame and add it to a sql database 
for row in df.itertuples():
    insert_sql_syntax = """
        INSERT INTO MOVIES(TITLE, YEAR, IMDB_RATINGS) 
        VALUES (?,?,?)
    """
    cursor.execute(insert_sql_syntax, row[1:])

db.commit()

## print the rows in the created database
for row in cursor.execute("""
    SELECT * FROM MOVIES
"""):
    print(row)
db.close()

