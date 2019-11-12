import requests
import pandas as pd
import sqlite3
import bs4
import re 

response = requests.get('https://www.imdb.com/chart/top?ref_=ft_250')
soup = bs4.BeautifulSoup(response.text, 'html.parser')

data = {
    "title":       [],
    "year":        [],
    "imdb_rating": []
}

list = soup.find("tbody", class_="lister-list").find_all("tr")
for row in list :
    data['title'].append(row.find(class_="titleColumn").find("a").text)
    years = row.find(class_="titleColumn").find(class_="secondaryInfo").text
    years = years.replace('(','')
    years = years.replace(')','')
    data['year'].append(years)
    data['imdb_rating'].append(row.find(class_="imdbRating").find("strong").text)

#print(data['title'])
#print(data['year'])
#print(data['imdb_rating'])


df = pd.DataFrame(data)

db     = sqlite3.connect(":memory:")
cursor = db.cursor()
cursor.execute("""
   CREATE TABLE MOVIES("TITLE" TEXT,
    "YEAR" INTEGER,
    "IMDB_RATINGS" REAL
    )
""")

for row in df.itertuples():
    insert_sql_syntax = """
        INSERT INTO MOVIES(TITLE, YEAR, IMDB_RATINGS) 
        VALUES (?,?,?)
    """
    cursor.execute(insert_sql_syntax, row[1:])
db.commit()


for row in cursor.execute("""
    SELECT * FROM MOVIES
"""):
    print(row)

    
db.close()
#print(data['title'])
#print(data['year'])
#print(data['imdb_rating'])