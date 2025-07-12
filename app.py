import streamlit as st
import pandas as pd
from ast import literal_eval
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import joblib

import warnings; warnings.simplefilter('ignore')

users = {
    "alice": 1,
    "bob": 5,
    "charlie": 9,
    "david": 12,
    "eva": 15
}

# ---- Load Data ----
@st.cache_data
def load_data():
    movies = pd.read_csv("./data/movies_metadata.csv")
    print(len(movies))
    ratings = pd.read_csv("./data/ratings.csv")

    # Clean genres
    movies['genres'] = movies['genres'].fillna('[]').apply(
        literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    print(len(movies['genres']))
    # Compute average ratings
    # vote_counts = ratings.groupby('movieId').count()['rating']
    # vote_averages = ratings.groupby('movieId').mean()['rating']

    # movies['vote_count'] = movies['id'].map(vote_counts)
    # movies['vote_average'] = movies['id'].map(vote_averages)

    # Filter movies with at least 50 votes
    movies = movies[movies['vote_count'] >= 10]
    print(len(movies))

    return movies


movies = load_data()
# print("Return : \n",movies.head())
placeholder_image = Image.open("./data/temp.jpeg")

# ---- Pages ----
def about_us():
    st.title("🎬 About Us")
    st.markdown("""
    Welcome to our **Movie Recommender System**!  
    This system is powered by movie metadata and user ratings, allowing you to discover the most popular and highest-rated films.

    **Dataset Overview:**
    - Over **10,000 movies**
    - Real user ratings from the **MovieLens dataset**
    - Rich metadata including **genres**, **overview**, **release date**, **language**, and more

    In this demo, we start with a **Simple Recommender** based on movie popularity and rating, with more advanced features coming soon.
    """)


def simple_recommender():
    st.title("🎥 Top Rated Movies")

    # Sort by rating
    top_movies = movies.sort_values(by=['vote_average', 'vote_count'], ascending=False).head(40)
    print(top_movies.columns)
    st.write("Debug message")
    for i in range(0, len(top_movies), 5):
        row = top_movies.iloc[i:i+5]
        cols = st.columns(5)
        for j, (_, movie) in enumerate(row.iterrows()):
            with cols[j]:
                poster_path = movie['poster_path']
                if pd.notna(poster_path) and poster_path.strip() != "":
                    image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    image_url = "temp.jpg"
                # poster_path = movie['imdb_id']
                # print(poster_path)
                st.image(image_url, use_column_width=True)
                st.markdown(f"**{movie['title']}**", unsafe_allow_html=True)
                # st.write(f"{movie['title']}")
    st.write('end')

def top_by_genre():
    st.title("🎬 Top Movies by Genre")

    # Initialize session state for show count
    if 'show_count' not in st.session_state:
        st.session_state.show_count = 20

    # Genre selection
    all_genres = set(genre for sublist in movies['genres'] for genre in sublist)
    genre = st.selectbox("Choose a Genre", sorted(all_genres))

    # Reset counter when genre changes
    if 'last_genre' not in st.session_state or st.session_state.last_genre != genre:
        st.session_state.show_count = 20
        st.session_state.last_genre = genre

    # Filter and sort
    genre_movies = movies[movies['genres'].apply(lambda x: genre in x)]
    top_movies = genre_movies.sort_values(by=['vote_average', 'vote_count'], ascending=False)

    # Limit number of movies shown
    visible_movies = top_movies.head(st.session_state.show_count)

    # Display in grid
    for i in range(0, len(visible_movies), 5):
        row = visible_movies.iloc[i:i+5]
        cols = st.columns(5)
        for j, (_, movie) in enumerate(row.iterrows()):
            with cols[j]:
                st.markdown('<div class="movie-tile">', unsafe_allow_html=True)

                # Poster image
                poster_path = movie['poster_path']
                if pd.notna(poster_path) and poster_path.strip() != "":
                    image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    image_url = "temp.jpg"

                st.image(image_url, use_column_width=True)
                st.markdown(f'<div class="movie-title"><strong>{movie["title"]}</strong></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Show more button
    if st.session_state.show_count < len(top_movies):
        if st.button("Show More"):
            st.session_state.show_count += 10

def content_based_recommender():
    st.title("🔍Content-Based Recommender")

    # Initialize session state
    if 'content_show_count' not in st.session_state:
        st.session_state.content_show_count = 20
    links_small = pd.read_csv('./data/links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    # md = movies.drop([19730, 29503, 35587])
    
    movies['id'] = movies['id'].astype('int')
    smd = movies[movies['id'].isin(links_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    # Movie selection
    movie_titles = smd['title'].sort_values().unique()
    selected_title = st.selectbox("Choose a Movie", movie_titles)

    # Reset count if title changed
    if 'last_content_title' not in st.session_state or st.session_state.last_content_title != selected_title:
        st.session_state.content_show_count = 20
        st.session_state.last_content_title = selected_title

    # Run your recommender
    idx = indices[selected_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    recs = smd.iloc[movie_indices][['title', 'poster_path']]

    # Limit based on "Show More"
    visible_recs = recs.head(st.session_state.content_show_count)

    # Display in grid
    for i in range(0, len(visible_recs), 5):
        row = visible_recs.iloc[i:i+5]
        cols = st.columns(len(row))
        for j, (_, movie) in enumerate(row.iterrows()):
            with cols[j]:
                poster_path = movie['poster_path']
                if pd.notna(poster_path) and poster_path.strip() != "":
                    image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    image_url = "temp.jpg"
                st.image(image_url, use_column_width=True)
                st.markdown(f"**{movie['title']}**")

    if st.session_state.content_show_count < len(recs):
        if st.button("Show More"):
            st.session_state.content_show_count += 10
@st.cache
def metadata_content():
    global movies
    st.title("🔍 Metadata Content-Based Recommender")
    # st.write("test")
    # Initialize session state
    if 'content_show_count' not in st.session_state:
        st.session_state.content_show_count = 20
    links_small = pd.read_csv('./data/links_small.csv')
    credits = pd.read_csv('./data/credits.csv')
    keywords = pd.read_csv('./data/keywords.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    movies['id'] = movies['id'].astype('int')
    movies = movies.merge(credits, on='id')
    movies = movies.merge(keywords, on='id')
    smd = movies[movies['id'].isin(links_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')
    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    smd['director'] = smd['crew'].apply(get_director)
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x,x, x])
    s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]
    stemmer = SnowballStemmer('english')
    stemmer.stem('dogs')
    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words
    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup']) 
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    # tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
    # tfidf_matrix = tf.fit_transform(smd['description'])
    # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # smd = smd.reset_index()
    # titles = smd['title']
    # indices = pd.Series(smd.index, index=smd['title'])
    # Movie selection
    movie_titles = smd['title'].sort_values().unique()
    selected_title = st.selectbox("Choose a Movie", movie_titles)

    # Reset count if title changed
    if 'last_content_title' not in st.session_state or st.session_state.last_content_title != selected_title:
        st.session_state.content_show_count = 20
        st.session_state.last_content_title = selected_title

    # Run your recommender
    idx = indices[selected_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    recs = smd.iloc[movie_indices][['title', 'poster_path']]

    # Limit based on "Show More"
    visible_recs = recs.head(st.session_state.content_show_count)

    # Display in grid
    for i in range(0, len(visible_recs), 5):
        row = visible_recs.iloc[i:i+5]
        cols = st.columns(len(row))
        for j, (_, movie) in enumerate(row.iterrows()):
            with cols[j]:
                poster_path = movie['poster_path']
                if pd.notna(poster_path) and poster_path.strip() != "":
                    image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    image_url = "temp.jpg"
                st.image(image_url, use_column_width=True)
                st.markdown(f"**{movie['title']}**")

    if st.session_state.content_show_count < len(recs):
        if st.button("Show More"):
            st.session_state.content_show_count += 10

def load_svd_model():
    return joblib.load('./svd_model.pkl')

# Recommend top N unseen movies for a user
def recommend_for_user(user_id, smd, ratings, svd_model, top_n=30):
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    unseen_movies = smd[~smd['id'].isin(rated_movie_ids)]

    predictions = []
    for _, movie in unseen_movies.iterrows():
        try:
            est = svd_model.predict(user_id, movie['id']).est
            predictions.append((movie['title'], movie['poster_path'], est))
        except:
            continue

    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions[:top_n]

def colab_filter():
    st.title("👥 Collaborative Filtering Recommender")

    ratings = pd.read_csv('./data/ratings_small.csv')
    smd = movies[movies['id'].isin(ratings)]
    # User Login Simulation
    username = st.selectbox("Login as:", list(users.keys()))
    user_id = users[username]
    st.markdown(f"Welcome, **{username}**! (User ID: `{user_id}`)")

    # Load trained SVD model
    with st.spinner("Loading model..."):
        svd = load_svd_model()

    # Get top recommendations
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_for_user(user_id, smd, ratings, svd, top_n=30)

    # Display in grid format
    st.subheader("🎬 Recommended Movies For You")

    for i in range(0, len(recommendations), 5):
        row = recommendations[i:i+5]
        cols = st.columns(len(row))
        for j, (title, poster_path, est_rating) in enumerate(row):
            with cols[j]:
                if pd.notna(poster_path) and poster_path.strip():
                    img_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    img_url = "temp.jpg"
                st.image(img_url, use_column_width=True)
                st.markdown(f"**{title}**")
                st.markdown(f"Predicted Rating: `{est_rating:.2f}`")

# ---- Navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Us", 
                                  "Simple Recommender",
                                  "Top by Genre",
                                  "Simple Content Recommender",
                                  "Metadata Content Recommender",
                                  "Collaborative Filtering"])

if page == "About Us":
    about_us()
elif page == "Simple Recommender":
    simple_recommender()
elif page == "Top by Genre":
    top_by_genre()
elif page == 'Simple Content Recommender':
    content_based_recommender()
elif page == 'Metadata Content Recommender':
    metadata_content()
elif page == "Collaborative Filtering":
    colab_filter()