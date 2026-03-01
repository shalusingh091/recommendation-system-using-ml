
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert genre text into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["genre"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title):
    idx = df[df["title"] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\nBecause you liked {movie_title}, you may also like:\n")

    for i in similarity_scores[1:4]:
        print(df.iloc[i[0]]["title"])

if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    recommend(movie)
