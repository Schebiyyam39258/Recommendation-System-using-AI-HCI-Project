# Importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the dataset from local file path
books = pd.read_csv("Books.csv", encoding='utf-8', on_bad_lines='skip')

# Selecting only book title and author columns and removing missing or duplicate entries
books = books[['title', 'authors']].dropna().drop_duplicates()

# Reducing the number of rows to improve speed and performance
books = books.head(3000)

# Creating a new column by combining book title and author for text analysis
books['content'] = books['title'].astype(str) + " " + books['authors'].astype(str)

# Converting text data into numerical values using TF-IDF technique
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books['content'])

# Measuring similarity between books using cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Creating an index to quickly find the book by title
indices = pd.Series(books.index, index=books['title']).drop_duplicates()

# Defining a function to recommend similar books based on input title
def recommend_books(title, num_recommendations=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    book_indices = [i[0] for i in sim_scores]
    return books['title'].iloc[book_indices].tolist()

# Showing some sample book titles for the user to choose from
print("Sample Book Titles from Dataset:\n")
print(books['title'].sample(10).to_string(index=False))

# Asking the user to enter a book title from the sample shown
print("\nEnter the exact book title from the list above to get recommendations:")
user_input = input("Your selected book: ")

# Providing recommendations if the entered book is found in the dataset
if user_input in indices:
    results = recommend_books(user_input)
    print(f"\nTop 5 Recommendations for '{user_input}':\n")
    for i, rec in enumerate(results, 1):
        print(f"{i}. {rec}")
else:
    print(f"\nBook '{user_input}' not found in the dataset. Please check the title.")
