import streamlit as st
import joblib

# Muat kembali fungsi-fungsi preprocessing dan model yang sudah disimpan sebelumnya
clean_punct = joblib.load('clean_punct_function.joblib')
tokenize_text = joblib.load('tokenize_text_function.joblib')
remove_stopwords = joblib.load('remove_stopwords_function.joblib')
tfidf_model = joblib.load('tfidf_model.joblib')
lda_model = joblib.load('lda_model.joblib')
knn_model = joblib.load('knn_model.joblib')

# Fungsi untuk melakukan prediksi pada teks baru
def predict_new_data(text):
    cleaned_data = clean_punct(text)
    tokenized_data = tokenize_text(cleaned_data)
    stopword_removed = remove_stopwords(tokenized_data)
    tfidf_new_data = tfidf_model.transform(stopword_removed)
    lda_new_data = lda_model.transform(tfidf_new_data)
    predicted = knn_model.predict(lda_new_data)
    return predicted

# Judul aplikasi dan deskripsi
st.title('Aplikasi Prediksi')
st.write('Masukkan teks untuk diprediksi kategori nya:')

# Input teks dari pengguna
user_input = st.text_area('Teks')

# Tombol prediksi
if st.button('Prediksi'):
    if user_input:
        prediction = predict_new_data([user_input])
        st.write('Hasil Prediksi:', prediction)
    else:
        st.write('Masukkan teks untuk melakukan prediksi.')
