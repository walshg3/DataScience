import requests
import pandas as pd
import sqlite3
import bs4
import re 


response      = requests.get('https://www.imdb.com/chart/top?ref_=ft_250')
soup          = bs4.BeautifulSoup(response.text, 'html.parser')

data = {
    "title":       [],
    "year":        [],
    "imdb_rating": []
}

movie_title   = soup.find_all('td' , class_ = 'titleColumn')  
movie_ratings = soup.find_all('td' , class_ = 'ratingColumn imdbRating')


for movie in movie_title:
    movie = movie.text.replace(' ','')
    movie = movie.replace('\n','')
    movie = movie[1 + movie.find("."): ]
    data['title'].append(re.sub(r'\(\d{4}\)','', movie))
    data['year'].append(re.findall(r"\((\d+)\)", movie))

for movie in movie_ratings:
    #print(movie.text)
    movie = movie.text.replace(' ','')
    movie = movie.replace('\n','')
    data['imdb_rating'].append(movie)


df = pd.DataFrame(data)
print(df)
#print(data['title'])
#print(data['year'])
#print(data['imdb_rating'])