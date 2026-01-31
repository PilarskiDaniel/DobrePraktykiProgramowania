import csv
from fastapi import FastAPI

app = FastAPI()

# =====================
# MODELE DANYCH
# =====================

class Movie:
    def __init__(self, movieId, title, genres):
        self.movieId = movieId
        self.title = title
        self.genres = genres


class Link:
    def __init__(self, movieId, imdbId, tmdbId):
        self.movieId = movieId
        self.imdbId = imdbId
        self.tmdbId = tmdbId


class Rating:
    def __init__(self, userId, movieId, rating, timestamp):
        self.userId = userId
        self.movieId = movieId
        self.rating = rating
        self.timestamp = timestamp


class Tag:
    def __init__(self, userId, movieId, tag, timestamp):
        self.userId = userId
        self.movieId = movieId
        self.tag = tag
        self.timestamp = timestamp


# =====================
# ENDPOINTY
# =====================

@app.get("/")
def hello():
    return {"hello": "world"}


@app.get("/movies")
def get_movies():
    movies = []

    with open("data/movies.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie = Movie(
                movieId=int(row["movieId"]),
                title=row["title"],
                genres=row["genres"]
            )
            movies.append(movie.__dict__)

    return movies


@app.get("/links")
def get_links():
    links = []

    with open("data/links.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            link = Link(
                movieId=int(row["movieId"]),
                imdbId=row["imdbId"],
                tmdbId=row["tmdbId"]
            )
            links.append(link.__dict__)

    return links


@app.get("/ratings")
def get_ratings():
    ratings = []

    with open("data/ratings.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rating = Rating(
                userId=int(row["userId"]),
                movieId=int(row["movieId"]),
                rating=float(row["rating"]),
                timestamp=int(row["timestamp"])
            )
            ratings.append(rating.__dict__)

    return ratings


@app.get("/tags")
def get_tags():
    tags = []

    with open("data/tags.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            tag = Tag(
                userId=int(row["userId"]),
                movieId=int(row["movieId"]),
                tag=row["tag"],
                timestamp=int(row["timestamp"])
            )
            tags.append(tag.__dict__)

    return tags

