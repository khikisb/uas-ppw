import streamlit as st
import joblib
import re

# Load model dan fungsi-fungsi preprocessing
tokenize_text = joblib.load('model/tokenize_text_function.joblib')
remove_stopwords = joblib.load('model/remove_stopwords_function.joblib')
tfidf_model = joblib.load('model/tfidf_model.joblib')  # Sesuaikan dengan model yang digunakan
lda_model = joblib.load('model/lda_model.joblib')  # Sesuaikan dengan model yang digunakan
knn_model = joblib.load('model/knn_model.joblib')  # Sesuaikan dengan model yang digunakan

# Fungsi untuk membersihkan tanda baca
def clean_punct(text):
    clean_tag = re.compile('@\S+')
    clean_url = re.compile('https?:\/\/.*[\r\n]*')
    clean_hastag = re.compile('#\S+')
    clean_symbol = re.compile('[^a-zA-Z]')
    text = clean_tag.sub('', str(text))
    text = clean_url.sub('', text)
    text = clean_hastag.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    return text

st.title('Prediksi Label Teks')

# Input teks baru dari pengguna
new_data = st.text_area('Masukkan teks baru:', '')

if st.button('Prediksi'):
    if new_data:
        # Pra-Pemrosesan teks baru
        cleaned_data = clean_punct(new_data)
        tokenized_data = tokenize_text(cleaned_data)
        stopword_removed = remove_stopwords(tokenized_data)

        # Mengonversi setiap elemen dalam stopword_removed menjadi string
        stopword_removed_str = [' '.join(tokens) for tokens in stopword_removed]

        # Transformasi TF-IDF pada data baru
        tfidf_new_data = tfidf_model.transform(stopword_removed_str)

        # Transformasi LDA pada data baru
        lda_new_data = lda_model.transform(tfidf_new_data)

        # Prediksi dengan model KNN pada data baru yang telah diproses
        predicted = knn_model.predict(lda_new_data)

        # Tampilkan hasil prediksi
        st.write('Hasil Prediksi:', predicted)
    else:
        st.warning('Silakan masukkan teks untuk melakukan prediksi.')
