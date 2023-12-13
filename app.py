import streamlit as st
import joblib

# Memuat kembali fungsi-fungsi preprocessing
clean_punct = joblib.load('model/clean_punct_function.joblib')
tokenize_text = joblib.load('model/tokenize_text_function.joblib')
remove_stopwords = joblib.load('model/remove_stopwords_function.joblib')

# Memuat kembali model TF-IDF, LDA, dan KNN
tfidf_model = joblib.load('model/tfidf_model.joblib')
lda_model = joblib.load('model/lda_model.joblib')
knn_model = joblib.load('model/knn_model.joblib')

# Mendefinisikan aplikasi Streamlit
st.title('Aplikasi Prediksi Klasifikasi Berita')
user_input = st.text_area('Masukkan teks')

# Fungsi untuk melakukan prediksi pada teks baru
def predict_new_data(text):
    cleaned_data = clean_punct(text)
    tokenized_data = tokenize_text(cleaned_data)
    stopword_removed = remove_stopwords(tokenized_data)
    tfidf_new_data = tfidf_model.transform(stopword_removed)
    lda_new_data = lda_model.transform(tfidf_new_data)
    predicted = knn_model.predict(lda_new_data)
    return predicted

# Tombol prediksi
if st.button('Prediksi'):
    if user_input:
        prediction = predict_new_data([user_input])
        st.write('Hasil Prediksi:', prediction)
    else:
        st.write('Masukkan teks Berita untuk melakukan prediksi.')
