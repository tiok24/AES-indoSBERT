import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sentence_transformers import SentenceTransformer, util

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AES IndoSBERT Skripsi", layout="wide")

# --- FUNGSI LOAD DATA (DI-CACHE AGAR CEPAT) ---
@st.cache_data
def load_data():
    nama_file = 'dataset question & answer skripsi - Raw (1).xlsx'
    
    # Load Kunci Jawaban
    df_kunci = pd.read_excel(nama_file, sheet_name='Kunci Jawaban')
    df_kunci_clean = df_kunci[['Kode', 'Jawaban']].rename(columns={'Jawaban': 'Kunci_Jawaban'})
    
    # Load semua sheet siswa
    sheet_siswa = ['DPK (TKJ)', 'MPP (TKJ-Telkom)', 'MPP (PPL) 2', 'MPP (RPL)']
    df_list = []
    for sheet in sheet_siswa:
        df_temp = pd.read_excel(nama_file, sheet_name=sheet)
        df_temp = df_temp[['Kode', 'Pertanyaan', 'Jawaban', 'Nilai']]
        df_temp['Jurusan'] = sheet # Tambahkan penanda asal jurusan
        df_list.append(df_temp)
        
    df_all_siswa = pd.concat(df_list, ignore_index=True)
    df_master = pd.merge(df_all_siswa, df_kunci_clean, on='Kode', how='inner')
    df_master = df_master.dropna(subset=['Jawaban', 'Kunci_Jawaban', 'Nilai']).reset_index(drop=True)
    
    return df_master

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    # Catatan: Jika saat deploy di Streamlit Cloud terjadi 'Out of Memory', 
    # ganti menjadi 'firqaaa/indo-sentence-bert-base'
    return SentenceTransformer("denaya/indoSBERT-large")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigasi Sistem")
halaman = st.sidebar.radio("Pilih Halaman:", ["📊 Halaman Analisis", "📝 Halaman Uji Scoring AI"])

# Load dataset
try:
    df = load_data()
    data_tersedia = True
except Exception as e:
    data_tersedia = False
    st.error(f"Gagal memuat data. Pastikan file Excel ada di repositori GitHub. Error: {e}")

# ==============================================================================
# HALAMAN 1: ANALISIS & METODOLOGI
# ==============================================================================
if halaman == "📊 Halaman Analisis" and data_tersedia:
    st.title("📊 Analisis Data & Pipeline Metodologi")
    
    tab1, tab2, tab3 = st.tabs(["Data Understanding & EDA", "Preprocessing Terbaik", "Pipeline & Evaluasi Model"])
    
    # --- TAB 1: DATA UNDERSTANDING & EDA ---
    with tab1:
        st.header("1. Data Understanding")
        st.write(f"Dataset terdiri dari **{len(df)}** baris pasangan jawaban siswa dan kunci jawaban.")
        st.dataframe(df[['Jurusan', 'Pertanyaan', 'Jawaban', 'Kunci_Jawaban', 'Nilai']].head(10))
        
        st.header("2. Exploratory Data Analysis (EDA)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Nilai Siswa")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Nilai'], bins=10, kde=True, color='skyblue', ax=ax)
            ax.set_title("Histogram Nilai Dosen/Guru")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Distribusi Jumlah Kata Jawaban")
            df['Word_Count'] = df['Jawaban'].apply(lambda x: len(str(x).split()))
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Jurusan', y='Word_Count', data=df, palette='Set2', ax=ax2)
            ax2.set_title("Panjang Jawaban per Jurusan")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

    # --- TAB 2: PREPROCESSING ---
    with tab2:
        st.header("Rekomendasi Preprocessing untuk IndoSBERT")
        st.info("**Aturan Emas untuk Transformer (SBERT):** Hindari penggunaan *Stemming* (penyeragaman kata dasar) dan penghapusan *Stopword* (kata hubung). Model berbasis SBERT membutuhkan konteks kalimat yang utuh untuk menangkap makna semantik. Menghapus kata hubung justru dapat menurunkan akurasi model.")
        
        st.markdown("Tahapan yang diaplikasikan:")
        st.markdown("1. **Case Folding:** Mengubah semua huruf menjadi kecil.\n2. **Punctuation Removal:** Menghapus tanda baca.\n3. **Whitespace Removal:** Menghapus spasi ganda dan karakter *newline*.")
        
        st.code("""
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()
        """, language='python')
        
        # Contoh visualisasi preprocessing
        sampel_teks = df['Jawaban'].iloc[0]
        st.write("**Teks Asli:**", sampel_teks)
        teks_bersih = str(sampel_teks).lower()
        teks_bersih = re.sub(r'\s+', ' ', teks_bersih)
        teks_bersih = teks_bersih.translate(str.maketrans('', '', string.punctuation)).strip()
        st.write("**Setelah Preprocessing:**", teks_bersih)

    # --- TAB 3: PIPELINE & EVALUASI ---
    with tab3:
        st.header("Pipeline Fine-Tuning & Dataloader")
        st.markdown("Berikut adalah kode representasi *training* model yang dijalankan di Google Colab menggunakan `CosineSimilarityLoss`.")
        
        st.code("""
# 1. Dataloader Preparation
train_examples = []
for idx, row in df_train.iterrows():
    train_examples.append(InputExample(
        texts=[row['Clean_Siswa'], row['Clean_Kunci']], 
        label=row['Nilai_Normalisasi'] # Skala 0.0 - 1.0
    ))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# 2. Fine-Tuning Model denaya/indoSBERT-large
model = SentenceTransformer("denaya/indoSBERT-large")
train_loss = losses.CosineSimilarityLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100
)
        """, language='python')
        
        st.subheader("Visualisasi Matriks Evaluasi (Simulasi Hasil Colab)")
        # Karena kita tidak bisa training langsung di Streamlit, kita memvisualisasikan format hasilnya
        # Anda bisa mengganti angka-angka ini dengan hasil asli dari Colab Anda nanti
        metrik_data = pd.DataFrame({
            'Metrik': ['Mean Absolute Error (MAE)', 'Pearson Correlation (r)', 'Quadratic Weighted Kappa (QWK)'],
            'Skor': [0.45, 0.82, 0.78] 
        })
        
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        sns.barplot(x='Skor', y='Metrik', data=metrik_data, palette='viridis', ax=ax3)
        ax3.set_xlim(0, 1)
        ax3.set_title("Performa Model pada Data Uji")
        st.pyplot(fig3)

# ==============================================================================
# HALAMAN 2: UJI SCORING (INFERENCE)
# ==============================================================================
elif halaman == "📝 Halaman Uji Scoring AI":
    st.title("📝 Uji Automatic Essay Scoring")
    st.markdown("Masukkan Kunci Jawaban dan Jawaban Siswa untuk diprediksi nilainya menggunakan **IndoSBERT-Large**.")
    
    with st.spinner("Memuat Model AI..."):
        model = load_model()
        
    skala_nilai = st.number_input("Skala Nilai Maksimal", min_value=1.0, value=4.0, step=1.0)
    
    col_a, col_b = st.columns(2)
    with col_a:
        kunci = st.text_area("🔑 Kunci Jawaban", height=200)
    with col_b:
        jawaban = st.text_area("🧑‍🎓 Jawaban Siswa", height=200)
        
    if st.button("Hitung Skor AI", type="primary", use_container_width=True):
        if kunci and jawaban:
            # Preprocessing sederhana sama seperti training
            def bersihkan(t):
                t = str(t).lower()
                t = re.sub(r'\s+', ' ', t)
                return t.translate(str.maketrans('', '', string.punctuation)).strip()
                
            clean_kunci = bersihkan(kunci)
            clean_siswa = bersihkan(jawaban)
            
            # Ekstraksi Fitur dan Cosine Similarity
            emb_kunci = model.encode(clean_kunci, convert_to_tensor=True)
            emb_siswa = model.encode(clean_siswa, convert_to_tensor=True)
            sim_score = util.cos_sim(emb_kunci, emb_siswa).item()
            sim_score = max(0.0, sim_score)
            
            nilai_akhir = sim_score * skala_nilai
            
            st.divider()
            st.subheader("📊 Hasil Penilaian")
            m1, m2 = st.columns(2)
            m1.metric("Prediksi Nilai", f"{nilai_akhir:.2f} / {skala_nilai}")
            m2.metric("Cosine Similarity", f"{sim_score:.4f}")
            st.progress(sim_score, text=f"Tingkat Kemiripan Semantik: {sim_score*100:.1f}%")
        else:
            st.warning("Mohon isi kedua kolom teks terlebih dahulu.")
