# ==============================
# Sistem Rekomendasi Film Sederhana
# Content-Based Filtering
# ==============================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Dataset
movies = pd.read_csv("movies.csv")

# 2. Mengisi data genre kosong (jika ada)
movies['genres'] = movies['genres'].fillna('')

# 3. Mengubah genre menjadi vektor angka
vectorizer = CountVectorizer(
    tokenizer=lambda x: x.split('|'),
    token_pattern=None
)

genre_matrix = vectorizer.fit_transform(movies['genres'])

# 4. Menghitung cosine similarity antar film
similarity_matrix = cosine_similarity(genre_matrix)

# 5. Fungsi rekomendasi film
def rekomendasi_film(judul_film, jumlah_rekomendasi=5):
    if judul_film not in movies['title'].values:
        return "Film tidak ditemukan dalam dataset."

    # Ambil index film
    idx = movies[movies['title'] == judul_film].index[0]

    # Ambil nilai similarity
    similarity_scores = list(enumerate(similarity_matrix[idx]))

    # Urutkan berdasarkan similarity tertinggi
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Ambil film paling mirip (kecuali film itu sendiri)
    rekomendasi_index = similarity_scores[1:jumlah_rekomendasi+1]

    # Ambil judul film
    rekomendasi = movies['title'].iloc[[i[0] for i in rekomendasi_index]]

    return rekomendasi

# 6. Contoh penggunaan
film_favorit = "Toy Story (1995)"
hasil = rekomendasi_film(film_favorit)

print("Film favorit:", film_favorit)
print("Rekomendasi film:")
print(hasil.to_string(index=False))
