""" 
Netflix Data Analysis Project
Dylan Roche
20 October 2025

This project conducts an introductory data analysis on the "Netflix Titles" dataset, 
which contains information about movies, TV shows, actors, countries, and more available 
on Netflix.

The objective of this analysis is to explore and uncover insights from Netflix’s 
content library, including identifying different trends, such as the countries 
with the most content, determining the most popular movie and TV actors, 
and visualizing patterns across different attributes.

The project also involves key data processing steps such as data cleaning, handling 
outliers, feature engineering, and rendering meaningful visualizations using Python libraries 
like Pandas and Matplotlib.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_df(file_path):
    df = pd.read_csv(file_path)
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['rating'] = df['rating'].fillna('Unrated')
    df = df.dropna(subset=['duration', 'date_added'])
    df['year_added'] = df['date_added'].dt.year
    return df

def print_stats(df):
    print("\nFirst five rows:\n", df.head())
    print("\nLast five rows:\n", df.tail())
    print("\nDataset columns:\n", df.columns)
    print("\nDatatypes description:\n", df.dtypes)
    print("\nDataset shape:\n", df.shape)
    print("\nNumber of missing values:\n", df.isnull().sum())
    print("\nNumber of duplicate values:\n", df.duplicated().sum())
    print("\nRelease year stats:\n", df['release_year'].describe())
    print("\nContent stats:\n", df['type'].value_counts())
    print("\nRelease year counts:\n", df['release_year'].value_counts())
    print("\nRating stats:\n", df['rating'].value_counts())

def convert_to_min(row):
    if row['duration_unit'].lower().startswith('min'):
        return row['duration_num']
    elif row['duration_unit'].lower().startswith('season'):
        avg_episode_length = 35
        avg_episodes_per_season = 10
        return row['duration_num'] * avg_episode_length * avg_episodes_per_season
    else:
        return None

def analyze_netflix_data(netflix):
    # Content over the Years Bar Chart
    content_over_time = netflix.groupby(['year_added', 'type']).size().unstack()
    content_over_time.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Netflix Content Over the Years")
    plt.xlabel("Year Added")
    plt.ylabel("Number of Titles")
    plt.legend(title='Type')
    plt.tight_layout()
    plt.show()
    print(
    "\nWhile movies make up the majority of Netflix content, the number of TV shows has been steadily increasing over time, "
    "possibly driven by the popularity of serialized storytelling and shows like Stranger Things."
)


    # Popular Genres Horizontal Bar Chart
    genres = netflix['listed_in'].str.split(', ').explode()
    genre_counts = genres.value_counts()
    genre_counts.plot(kind='barh', color='skyblue', figsize=(10, 6))
    plt.title("Popular Genres")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre Title")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
   

    genres = netflix[['country', 'listed_in']].copy()
    genres['listed_in'] = genres['listed_in'].str.split(', ')
    genres = genres.explode('listed_in')

    top_countries = genres['country'].value_counts().head(10).index
    genres = genres[genres['country'].isin(top_countries)]

    genre_country = genres.pivot_table(index='listed_in', columns='country', aggfunc='size', fill_value=0)


    plt.figure(figsize=(12, 8))
    sns.heatmap(genre_country, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Genres by Top 10 Countries")
    plt.xlabel("Country")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.show()
    print("\n This heatmap reveals some powerful insights. For instance, dramas are the most popular form of content in the U.S., followed by comedies. "
    "Comedies, in particular, appear to perform well in both the U.S. and India, suggesting that this genre continues to attract strong global interest "
    "and could be leveraged to draw more viewers to the platform. In Japan, anime is especially popular, which aligns with its cultural significance "
    "and dominance in the country's entertainment industry.")


    # Most Popular Countries Horizontal Bar Chart
    most_popular_countries = netflix['country'].str.split(', ').explode()
    most_popular_countries = most_popular_countries.value_counts().head(10)
    most_popular_countries.plot(kind='barh', color='skyblue', figsize=(10, 6))
    plt.title("Top 10 Countries that Produce Content")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Split by type
    netflix_movies = netflix[netflix['type'] == 'Movie'].copy()
    netflix_tv = netflix[netflix['type'] == 'TV Show'].copy()

    print("Movies:\n", netflix_movies.head(10))
    print("TV Shows:\n", netflix_tv.head(10))

    print(f"Number of Movies: {len(netflix_movies)}")
    print(f"Number of Shows: {len(netflix_tv)}")

    print(f"Average Netflix movie length: {netflix_movies['duration_minutes'].mean():.2f} mins")
    print(f"Median Netflix movie length: {netflix_movies['duration_minutes'].median():.2f} mins")
    print(f"Max Netflix movie length: {netflix_movies['duration_minutes'].max()} mins")
    print(f"Min Netflix movie length: {netflix_movies['duration_minutes'].min()} mins")
    print(f"Std dev movie length: {netflix_movies['duration_minutes'].std():.2f} mins")

    print(f"Average Netflix show length: {netflix_tv['duration_minutes'].mean():.2f} mins")
    print(f"Median Netflix show length: {netflix_tv['duration_minutes'].median():.2f} mins")
    print(f"Max Netflix show length: {netflix_tv['duration_minutes'].max()} mins")
    print(f"Min Netflix show length: {netflix_tv['duration_minutes'].min()} mins")
    print(f"Std dev show length: {netflix_tv['duration_minutes'].std():.2f} mins")

    sns.histplot(data=netflix_movies, x='duration_minutes', kde=True)
    plt.title("Distribution of Netflix Movie Runtimes")
    plt.xlabel("Runtime (mins)")
    plt.tight_layout()
    plt.show()
    print("\nThis histogram, which includes a Kernel Density Estimate (KDE) curve, reveals a clear peak around the "
      "90–100 minute mark. Although longer films (around 120 minutes) still attract viewers, the data suggests that "
      "the optimal Netflix movie length falls between 90 and 100 minutes -- a duration that balances engagement and accessibility.")


    sns.histplot(data=netflix_tv, x='duration_minutes', kde=True)
    plt.title("Distribution of Netflix TV Show Runtimes")
    plt.xlabel("Runtime (mins)")
    plt.tight_layout()
    plt.show()
   

    tv_show_seasons = netflix_tv[netflix_tv['duration'].str.contains('Season', case=False, na=False)]
    season_counts = tv_show_seasons['duration_num'].value_counts().sort_index()
    season_counts.plot(kind='bar', color='purple', figsize=(10, 6))
    plt.title("Number of TV Shows by Season Count")
    plt.xlabel("Number of Seasons")
    plt.ylabel("Number of TV Shows")
    plt.tight_layout()
    plt.show()

    correlation_matrix = netflix[['duration_minutes', 'release_year', 'year_added']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix of Content Duration, Release Year, Year Added")
    plt.tight_layout()
    plt.show()
    print("\nThis correlation matrix suggests that there are weak correlations between a title's duration, "
      "release year, and the year it was added to Netflix. This indicates that durations have remained relatively stable "
      "over time, and that Netflix adds both recently released and older content rather than focusing solely on new releases.")
    
    # IQR 
    q1_movies = netflix_movies['duration_minutes'].quantile(0.25)
    q3_movies = netflix_movies['duration_minutes'].quantile(0.75)
    iqr_movies = q3_movies - q1_movies
    lower_bound_movies = q1_movies - 1.5 * iqr_movies
    upper_bound_movies = q3_movies + 1.5 * iqr_movies

    outliers_movies = netflix_movies[
        (netflix_movies['duration_minutes'] < lower_bound_movies) |
        (netflix_movies['duration_minutes'] > upper_bound_movies)
    ]

    # Outlier detection for TV shows
    q1_tv = netflix_tv['duration_minutes'].quantile(0.25)
    q3_tv = netflix_tv['duration_minutes'].quantile(0.75)
    iqr_tv = q3_tv - q1_tv
    lower_bound_tv = q1_tv - 1.5 * iqr_tv
    upper_bound_tv = q3_tv + 1.5 * iqr_tv

    outliers_tv = netflix_tv[
        (netflix_tv['duration_minutes'] < lower_bound_tv) |
        (netflix_tv['duration_minutes'] > upper_bound_tv)
    ]

    print(f"Number of Outliers in Movies: {len(outliers_movies)}")
    print(f"Number of Outliers in Shows: {len(outliers_tv)}")

    clean_movies = netflix_movies[
        (netflix_movies['duration_minutes'] >= 30) &
        (netflix_movies['duration_minutes'] <= 300)
    ]
    clean_movies = clean_movies[clean_movies['cast'].str.lower() != 'unknown']

    clean_tv = netflix_tv[
        (netflix_tv['duration_minutes'] >= 30) &
        (netflix_tv['duration_minutes'] <= 2000)
    ]
    clean_tv = clean_tv[clean_tv['cast'].str.lower() != 'unknown']

    sns.boxplot(data=clean_movies, x='rating', y='duration_minutes')
    plt.title("Ratings vs. Durations Boxplot (Movies)")
    plt.tight_layout()
    plt.show()

    sns.boxplot(data=clean_tv, x='rating', y='duration_minutes')
    plt.title("Ratings vs. Durations Boxplot (TV Shows)")
    plt.tight_layout()
    plt.show()

    movies_per_year = clean_movies.groupby('year_added').size()
    shows_per_year = clean_tv.groupby('year_added').size()
    plt.figure(figsize=(12, 6))
    plt.plot(movies_per_year.index, movies_per_year.values, marker='o', linestyle='-', label='Movies', color='blue')
    plt.plot(shows_per_year.index, shows_per_year.values, marker='o', linestyle='-', label='TV Shows', color='orange')
    plt.title("Number of Movies and Shows Added over Time")
    plt.xlabel("Year Added")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nThis line chart demonstrates and reaffirms certain insights -- most notably, that movies dominate Netflix’s content library, "
      "with more films being added to the platform than TV shows. From 2015 to 2019, both categories saw growth, with a particularly sharp increase in movies. "
      "However, since 2020, the number of new movies and shows added to Netflix has trended downward, likely due to factors such as the COVID-19 pandemic, "
      "rising production costs, and shifting market conditions.")

    # Movie actors analysis
    clean_movies['cast_split'] = clean_movies['cast'].str.split(', ')
    movie_actors = clean_movies.explode('cast_split')[['cast_split', 'country']].copy()
    movie_actors.rename(columns={'cast_split': 'actor'}, inplace=True)
    movie_actors = movie_actors[
        (movie_actors['actor'].str.lower() != 'unknown') &
        (movie_actors['country'].str.lower() != 'unknown')
    ]
    popular_movie_actors = movie_actors['actor'].value_counts().head(10)
    top_actor_country_movies = movie_actors.groupby(['actor', 'country']).size().sort_values(ascending=False).head(10)
    top_actor_country_movies.index = [f"{actor} ({country})" for actor, country in top_actor_country_movies.index]
    top_actor_country_movies.plot(kind='bar', color='blue', figsize=(12, 6))
    plt.title("Top 10 Movie Actors and Their Country of Origin")
    plt.xlabel("Actor (Country)")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    print("\nThis bar chart shows that the top 10 actors by number of movies are all from India. "
      "This reflects the popularity of Bollywood films on Netflix, "
      "suggesting that regional content production influences actor visibility as well as "
      "engagement in certain markets.")


    # TV actors analysis
    clean_tv['cast_split'] = clean_tv['cast'].str.split(', ')
    tv_actors = clean_tv.explode('cast_split')[['cast_split', 'country']].copy()
    tv_actors.rename(columns={'cast_split': 'actor'}, inplace=True)
    tv_actors = tv_actors[
        (tv_actors['actor'].str.lower() != 'unknown') &
        (tv_actors['country'].str.lower() != 'unknown')
    ]
    popular_tv_actors = tv_actors['actor'].value_counts().head(10)
    top_actor_country_tv = tv_actors.groupby(['actor', 'country']).size().sort_values(ascending=False).head(10)
    top_actor_country_tv.index = [f"{actor} ({country})" for actor, country in top_actor_country_tv.index]
    top_actor_country_tv.plot(kind='bar', color='green', figsize=(12, 6))
    plt.title("Top 10 TV Actors and Their Country of Origin")
    plt.xlabel("Actor (Country)")
    plt.ylabel("Number of TV Shows")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    print("\nThis bar chart shows that the top 10 TV actors by number of shows are all from Japan. "
      "These actors are primarily voice actors in anime, which dominates the content in the country. "
      "Additionally, many of the same voice actors contribute to multiple projects, which explains why they appear in the top 10 repeatedly.")

    # Print and plot popular actors
    print("\nTop 10 Popular Movie Actors:\n", popular_movie_actors)
    popular_movie_actors.plot(kind='bar', color='blue', figsize=(10, 5))
    plt.title("Top 10 Popular Movie Actors")
    plt.xlabel("Actor")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\nTop 10 Popular TV Actors:\n", popular_tv_actors)
    popular_tv_actors.plot(kind='bar', color='green', figsize=(10, 5))
    plt.title("Top 10 Popular TV Actors")
    plt.xlabel("Actor")
    plt.ylabel("Number of TV Shows")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "c:/Users/User/Downloads/archive (7)/netflix_titles.csv"
    netflix = load_and_clean_df(file_path)

    # Convert duration column to string
    netflix['duration'] = netflix['duration'].astype(str)

    # Extract number and unit using str.replace alternative
    netflix['duration_num'] = netflix['duration'].str.extract('(\d+)').astype(float)
    netflix['duration_unit'] = netflix['duration'].str.replace(r'[\d ]', '', regex=True).replace('', 'min').str.strip()

    # Calculate duration in minutes
    netflix['duration_minutes'] = netflix.apply(convert_to_min, axis=1)

    print_stats(netflix)
    analyze_netflix_data(netflix)
