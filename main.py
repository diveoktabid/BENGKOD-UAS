import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Import untuk visualisasi
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Tingkat Obesitas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Embedded (sebagai backup jika file external tidak terbaca)
def get_embedded_css():
    """CSS embedded sebagai fallback"""
    return """
    <style>
    /* Header utama */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
        color: white !important;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }

    .section-header h3 {
        margin: 0;
        color: #007bff !important;
        font-weight: bold;
    }

    /* Result styling */
    .result-success {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #155724 !important;
    }

    .result-success h2, .result-success h3, .result-success h4, .result-success h5,
    .result-success p, .result-success div {
        color: #155724 !important;
    }

    .result-warning {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #856404 !important;
    }

    .result-warning h2, .result-warning h3, .result-warning h4, .result-warning h5,
    .result-warning p, .result-warning div {
        color: #856404 !important;
    }

    .result-danger {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        color: #721c24 !important;
    }

    .result-danger h2, .result-danger h3, .result-danger h4, .result-danger h5,
    .result-danger p, .result-danger div {
        color: #721c24 !important;
    }

    /* Info cards */
    .info-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        color: #212529 !important;
    }

    .info-card h4, .info-card h5 {
        color: #007bff !important;
        font-weight: bold;
    }

    .info-card p, .info-card li, .info-card strong {
        color: #212529 !important;
        line-height: 1.6;
    }

    .info-card table td {
        color: #212529 !important;
    }

    /* Photo placeholder */
    .photo-placeholder {
        width: 150px;
        height: 150px;
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        margin: 0 auto 1rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        box-sizing: border-box;
    }

    .photo-placeholder p {
        color: #6c757d !important;
        text-align: center;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .photo-placeholder code {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 0.8rem;
        word-break: break-all;
    }

    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        color: #212529 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .metric-card h5 {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #007bff !important;
    }

    .metric-card p, .metric-card strong {
        color: #212529 !important;
        margin: 0.3rem 0;
    }

    /* Text kontras */
    .stMarkdown p {
        color: #212529 !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #007bff !important;
    }

    .stMarkdown ul li, .stMarkdown ol li, .stMarkdown strong {
        color: #212529 !important;
    }

    .stMarkdown code {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Sidebar styling */
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown strong {
        color: #212529 !important;
    }

    .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: #007bff !important;
    }

    /* Form styling */
    .stSelectbox label, .stSlider label {
        color: #212529 !important;
        font-weight: 500;
    }

    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stForm .stButton > button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 1rem 2rem;
        font-weight: bold;
        font-size: 1.2rem;
        width: 100%;
        margin-top: 1rem;
    }

    .stForm .stButton > button:hover {
        background: linear-gradient(90deg, #218838 0%, #1ab085 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
    }

    /* DataFrame styling */
    .stDataFrame table {
        color: #212529 !important;
    }

    .stDataFrame th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        font-weight: bold;
    }

    .stDataFrame td {
        color: #212529 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 8px;
        color: #495057 !important;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white !important;
    }

    /* Metric container styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="metric-container"] label {
        color: #212529 !important;
        font-weight: bold;
    }

    div[data-testid="metric-container"] div {
        color: #007bff !important;
        font-weight: bold;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .info-card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 0.75rem;
        }
        
        .photo-placeholder {
            width: 120px;
            height: 120px;
        }
        
        .photo-placeholder p {
            font-size: 0.8rem;
        }
    }
    </style>
    """

# Load CSS dengan fallback
def load_css():
    """Load CSS dari file external atau gunakan embedded CSS"""
    try:
        # Coba load dari file external
        if os.path.exists('styles.css'):
            with open('styles.css', 'r', encoding='utf-8') as f:
                css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
            st.success("CSS eksternal berhasil dimuat dari styles.css")
        else:
            raise FileNotFoundError("File styles.css tidak ditemukan")
    except Exception as e:
        # Gunakan CSS embedded sebagai fallback
        st.warning(f"Tidak dapat memuat styles.css ({str(e)}). Menggunakan CSS embedded.")
        st.markdown(get_embedded_css(), unsafe_allow_html=True)

# Fungsi untuk load model
@st.cache_resource
def load_models():
    """Load model machine learning dan preprocessing"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        try:
            with open('models/feature_columns.json', 'r') as f:
                feature_columns = json.load(f)
        except:
            feature_columns = [
                'Age', 'Height', 'Weight', 'BMI', 
                'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 
                'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                'Gender_encoded', 'CAEC_encoded', 'CALC_encoded', 'MTRANS_encoded'
            ]
            
        return model, scaler, label_encoder, feature_columns
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None, None, None

def main():
    """Fungsi utama aplikasi"""
    # Load CSS dengan fallback
    load_css()
    
    # Header yang menarik
    st.markdown("""
    <div class='main-header'>
        <h1>Sistem Prediksi Tingkat Obesitas</h1>
        <p>Aplikasi Machine Learning untuk Analisis dan Prediksi Kesehatan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk navigasi dan info
    setup_sidebar()
    
    # Load models
    model, scaler, label_encoder, feature_columns = load_models()
    
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file model tersedia di folder 'models/'")
        return
    
    # Tabs untuk navigasi konten
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediksi Obesitas", 
        "Profil Mahasiswa", 
        "Informasi Model", 
        "Visualisasi Data"
    ])
    
    with tab1:
        prediction_tab(model, scaler, label_encoder, feature_columns)
    
    with tab2:
        profile_tab()
    
    with tab3:
        model_info_tab()
    
    with tab4:
        visualization_tab()

def setup_sidebar():
    """Setup sidebar dengan informasi aplikasi"""
    with st.sidebar:
        st.markdown("### Navigasi")
        st.markdown("Pilih menu di bawah untuk menjelajahi aplikasi:")
        
        # Profile section
        st.markdown("---")
        st.markdown("### Profil Mahasiswa")
        st.markdown("""
        **Nama:** Dive Oktabid Fikhri  
        **NIM:** A11.2022.14202  
        **Program Studi:** Teknik Informatika  
        **Universitas:** Dian Nuswantoro  
        **Mata Kuliah:** Bengkel Koding  
        **Semester:** Genap 2024/2025
        """)
        
        st.markdown("---")
        st.markdown("### Performa Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi", "96.2%")
            st.metric("Presisi", "96.1%")
        with col2:
            st.metric("Recall", "96.2%")
            st.metric("F1-Score", "96.1%")
        
        st.markdown("---")
        st.markdown("### Info Dataset")
        st.markdown("""
        **Total Sampel:** 2,111 orang  
        **Jumlah Fitur:** 17 variabel  
        **Target Kelas:** 7 tingkat obesitas  
        **Sumber Data:** Meksiko, Peru, Kolombia  
        **Model Terbaik:** Random Forest
        """)

def prediction_tab(model, scaler, label_encoder, feature_columns):
    """Tab untuk prediksi obesitas"""
    st.markdown("""
    <div class='section-header'>
        <h3>Form Input Data Kesehatan</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Silakan isi formulir di bawah ini dengan data kesehatan Anda. 
    Semua informasi akan digunakan untuk memprediksi tingkat obesitas dengan akurasi tinggi.
    """)
    
    with st.form("form_prediksi_obesitas"):
        # Bagian 1: Data Pribadi
        st.markdown("### Data Pribadi")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Jenis Kelamin",
                ["Laki-laki", "Perempuan"],
                help="Pilih jenis kelamin Anda"
            )
            age = st.slider(
                "Usia (tahun)", 
                min_value=14, max_value=70, value=25, step=1,
                help="Masukkan usia Anda dalam tahun"
            )
        
        with col2:
            height = st.slider(
                "Tinggi Badan (cm)", 
                min_value=140, max_value=200, value=170, step=1,
                help="Masukkan tinggi badan Anda dalam centimeter"
            )
            weight = st.slider(
                "Berat Badan (kg)", 
                min_value=35, max_value=180, value=70, step=1,
                help="Masukkan berat badan Anda dalam kilogram"
            )
        
        # Convert height to meters for calculation
        height_m = height / 100.0
        
        st.markdown("---")
        
        # Bagian 2: Riwayat Keluarga dan Kebiasaan
        st.markdown("### Riwayat Keluarga & Kebiasaan Hidup")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            family_history = st.selectbox(
                "Riwayat Obesitas Keluarga",
                ["Tidak Ada", "Ada"],
                help="Apakah ada anggota keluarga yang mengalami obesitas?"
            )
        
        with col2:
            smoke = st.selectbox(
                "Status Merokok",
                ["Tidak Merokok", "Merokok"],
                help="Apakah Anda merokok?"
            )
        
        with col3:
            scc = st.selectbox(
                "Pemantauan Kalori",
                ["Tidak Memantau", "Memantau"],
                help="Apakah Anda memantau asupan kalori harian?"
            )
        
        st.markdown("---")
        
        # Bagian 3: Pola Makan
        st.markdown("### Pola Makan dan Konsumsi")
        col1, col2 = st.columns(2)
        
        with col1:
            favc = st.selectbox(
                "Konsumsi Makanan Berkalori Tinggi",
                ["Jarang", "Sering"],
                help="Seberapa sering Anda mengonsumsi makanan berkalori tinggi?"
            )
            
            fcvc = st.select_slider(
                "Frekuensi Konsumsi Sayuran",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: {
                    1: "1 - Sangat Jarang",
                    2: "2 - Jarang", 
                    3: "3 - Kadang-kadang",
                    4: "4 - Sering",
                    5: "5 - Sangat Sering"
                }[x],
                help="Seberapa sering Anda makan sayuran dalam seminggu?"
            )
            
            ncp = st.select_slider(
                "Jumlah Makan Utama per Hari",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x} kali sehari",
                help="Berapa kali Anda makan besar dalam sehari?"
            )
        
        with col2:
            caec = st.selectbox(
                "Kebiasaan Mengemil",
                ["Tidak Pernah", "Kadang-kadang", "Sering", "Selalu"],
                help="Seberapa sering Anda makan camilan di antara waktu makan utama?"
            )
            
            ch2o = st.select_slider(
                "Konsumsi Air per Hari (Liter)",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: {
                    1: "1 - Kurang dari 1L",
                    2: "2 - 1-2 Liter", 
                    3: "3 - 2-3 Liter",
                    4: "4 - 3-4 Liter",
                    5: "5 - Lebih dari 4L"
                }[x],
                help="Berapa liter air yang Anda minum per hari?"
            )
            
            calc = st.selectbox(
                "Konsumsi Alkohol",
                ["Tidak Pernah", "Kadang-kadang", "Sering", "Selalu"],
                help="Seberapa sering Anda mengonsumsi minuman beralkohol?"
            )
        
        st.markdown("---")
        
        # Bagian 4: Aktivitas Fisik
        st.markdown("### Aktivitas Fisik dan Gaya Hidup")
        col1, col2 = st.columns(2)
        
        with col1:
            faf = st.select_slider(
                "Frekuensi Aktivitas Fisik/Olahraga",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: {
                    1: "1 - Tidak Pernah",
                    2: "2 - Jarang (1-2x/minggu)", 
                    3: "3 - Kadang (3-4x/minggu)",
                    4: "4 - Sering (5-6x/minggu)",
                    5: "5 - Sangat Sering (Setiap hari)"
                }[x],
                help="Seberapa sering Anda melakukan aktivitas fisik atau olahraga?"
            )
            
            tue = st.select_slider(
                "Waktu Penggunaan Teknologi per Hari",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: {
                    1: "1 - Kurang dari 2 jam",
                    2: "2 - 2-4 jam", 
                    3: "3 - 4-6 jam",
                    4: "4 - 6-8 jam",
                    5: "5 - Lebih dari 8 jam"
                }[x],
                help="Berapa lama Anda menggunakan gadget/teknologi per hari?"
            )
        
        with col2:
            mtrans = st.selectbox(
                "Moda Transportasi Utama",
                ["Jalan Kaki", "Sepeda", "Transportasi Umum", "Motor", "Mobil Pribadi"],
                help="Apa moda transportasi yang paling sering Anda gunakan?"
            )
        
        st.markdown("---")
        
        # Tombol prediksi
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "Prediksi Tingkat Obesitas Saya",
                use_container_width=True,
                help="Klik untuk mendapatkan prediksi tingkat obesitas berdasarkan data yang Anda masukkan"
            )
        
        if submitted:
            with st.spinner("Sedang menganalisis data kesehatan Anda..."):
                # Proses prediksi
                process_prediction(
                    gender, age, height_m, weight, family_history, favc, fcvc, ncp,
                    caec, smoke, ch2o, scc, faf, tue, calc, mtrans,
                    model, scaler, label_encoder, feature_columns
                )

def process_prediction(gender, age, height_m, weight, family_history, favc, fcvc, ncp,
                      caec, smoke, ch2o, scc, faf, tue, calc, mtrans,
                      model, scaler, label_encoder, feature_columns):
    """Proses prediksi obesitas"""
    # Konversi input ke format yang dibutuhkan model
    gender_model = "Male" if gender == "Laki-laki" else "Female"
    family_history_model = "yes" if family_history == "Ada" else "no"
    favc_model = "yes" if favc == "Sering" else "no"
    smoke_model = "yes" if smoke == "Merokok" else "no"
    scc_model = "yes" if scc == "Memantau" else "no"
    
    # Mapping untuk categorical variables
    caec_mapping = {
        "Tidak Pernah": "no",
        "Kadang-kadang": "Sometimes", 
        "Sering": "Frequently",
        "Selalu": "Always"
    }
    
    calc_mapping = {
        "Tidak Pernah": "no",
        "Kadang-kadang": "Sometimes",
        "Sering": "Frequently", 
        "Selalu": "Always"
    }
    
    mtrans_mapping = {
        "Jalan Kaki": "Walking",
        "Sepeda": "Bike",
        "Transportasi Umum": "Public_Transportation",
        "Motor": "Motorbike",
        "Mobil Pribadi": "Automobile"
    }
    
    # Persiapan data
    input_data = prepare_input_data(
        gender_model, age, height_m, weight, family_history_model, 
        favc_model, fcvc, ncp, caec_mapping[caec], smoke_model, 
        ch2o, scc_model, faf, tue, calc_mapping[calc], 
        mtrans_mapping[mtrans], feature_columns
    )
    
    if input_data is not None:
        # Prediksi
        prediction, probabilities = make_prediction(model, scaler, input_data)
        
        if prediction is not None:
            # Tampilkan hasil
            display_results(prediction, probabilities, label_encoder)
            
            # Analisis kesehatan
            bmi = weight / (height_m ** 2)
            display_health_analysis(bmi, age, faf, favc_model, weight, height_m * 100, gender)

def profile_tab():
    """Tab untuk profil mahasiswa"""
    st.markdown("""
    <div class='section-header'>
        <h3>Profil Mahasiswa</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Profile card dengan foto
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class='info-card text-center'>
            <div class='photo-placeholder'>
                <p style='margin: 0; font-size: 0.9rem; color: #6c757d; line-height: 1.4;'>
                    Foto Mahasiswa<br>
                    <strong>Masukkan link foto Anda di sini:</strong><br>
                    <code style='font-size: 0.8rem; background: #f8f9fa; padding: 2px 4px; border-radius: 3px; color: #495057;'>
                        https://link-foto-anda.com/foto.jpg
                    </code>
                </p>
            </div>
            <h4 style='margin: 1rem 0 0 0;'>Dive Oktabid Fikhri</h4>
            <p style='margin: 0.5rem 0 0 0; color: #6c757d;'>A11.2022.14202</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='margin-top: 0;'>Informasi Akademik</h4>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Nama Lengkap</td><td style='padding: 0.5rem 0; color: #212529;'>Dive Oktabid Fikhri</td></tr>
                <tr style='background-color: #f8f9fa;'><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>NIM</td><td style='padding: 0.5rem 0; color: #212529;'>A11.2022.14202</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Program Studi</td><td style='padding: 0.5rem 0; color: #212529;'>Teknik Informatika</td></tr>
                <tr style='background-color: #f8f9fa;'><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Universitas</td><td style='padding: 0.5rem 0; color: #212529;'>Dian Nuswantoro</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Mata Kuliah</td><td style='padding: 0.5rem 0; color: #212529;'>Bengkel Koding</td></tr>
                <tr style='background-color: #f8f9fa;'><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Semester</td><td style='padding: 0.5rem 0; color: #212529;'>Genap 2024/2025</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: bold; color: #495057;'>Dosen Pengampu</td><td style='padding: 0.5rem 0; color: #212529;'>Tim Data Science</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    <div class='info-card'>
        <h4 style='margin-top: 0;'>Ringkasan Project Capstone</h4>
        <p><strong>Judul:</strong> Sistem Prediksi Tingkat Obesitas Menggunakan Machine Learning</p>
        <p><strong>Tujuan:</strong> Mengembangkan aplikasi web untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan dan aktivitas fisik</p>
        <p><strong>Teknologi:</strong> Python, Scikit-learn, Streamlit, Pandas, NumPy</p>
        <p><strong>Dataset:</strong> 2,111 sampel dari Meksiko, Peru, dan Kolombia dengan 17 fitur kesehatan</p>
        <p><strong>Model Terbaik:</strong> Random Forest dengan akurasi 96.2%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Achievements
    st.markdown("### Pencapaian Project")
    
    achievements = [
        {"title": "Akurasi Tinggi", "desc": "Mencapai akurasi 96.2% dengan model Random Forest"},
        {"title": "Analisis Mendalam", "desc": "EDA komprehensif dengan 10 visualisasi data"},
        {"title": "Optimasi Model", "desc": "Hyperparameter tuning untuk performa maksimal"},
        {"title": "Aplikasi Web", "desc": "Deployment aplikasi interaktif dengan Streamlit"},
        {"title": "Visualisasi Data", "desc": "Dashboard informatif dengan grafik interaktif"},
        {"title": "UI/UX Friendly", "desc": "Interface yang mudah digunakan dan responsive"}
    ]
    
    cols = st.columns(3)
    for i, achievement in enumerate(achievements):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card'>
                <h5 style='margin: 0;'>{achievement['title']}</h5>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{achievement['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def model_info_tab():
    """Tab untuk informasi model"""
    st.markdown("""
    <div class='section-header'>
        <h3>Informasi Model Machine Learning</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance
    st.markdown("### Performa Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h5 style='margin-top: 0;'>Model Terbaik: Random Forest</h5>
        </div>
        """, unsafe_allow_html=True)
        
        performance_data = {
            'Metrik Evaluasi': ['Akurasi', 'Presisi', 'Recall', 'F1-Score', 'Cross-Validation'],
            'Skor (%)': [96.2, 96.1, 96.2, 96.1, 95.8]
        }
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h5 style='margin-top: 0;'>Perbandingan Semua Model</h5>
        </div>
        """, unsafe_allow_html=True)
        
        comparison_data = {
            'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM', 'KNN'],
            'Akurasi (%)': [96.2, 95.8, 94.3, 93.9, 92.1],
            'F1-Score (%)': [96.1, 95.7, 94.2, 93.8, 92.0]
        }
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
    
    # Feature importance
    st.markdown("### Tingkat Kepentingan Fitur")
    
    importance_data = {
        'Fitur': ['BMI (Indeks Massa Tubuh)', 'Berat Badan', 'Tinggi Badan', 'Usia', 
                 'Riwayat Keluarga', 'Aktivitas Fisik', 'Konsumsi Air', 'Jenis Kelamin'],
        'Tingkat Kepentingan (%)': [34.2, 29.8, 15.6, 8.9, 6.7, 3.1, 1.7, 1.0],
        'Kategori': ['Fisik', 'Fisik', 'Fisik', 'Demografis', 'Genetik', 'Gaya Hidup', 'Gaya Hidup', 'Demografis']
    }
    df_importance = pd.DataFrame(importance_data)
    st.dataframe(df_importance, use_container_width=True)
    
    # Key insights
    st.markdown("### Temuan Penting")
    
    insights = [
        "BMI adalah prediktor terkuat dengan kontribusi 34.2% terhadap prediksi",
        "Faktor fisik (BMI, Berat, Tinggi) menyumbang sekitar 80% akurasi prediksi",
        "Riwayat keluarga memiliki pengaruh genetik yang signifikan (6.7%)",
        "Aktivitas fisik dan gaya hidup berkontribusi pada prediksi obesitas",
        "Model berhasil mengidentifikasi faktor-faktor medis yang relevan"
    ]
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"**{i}.** {insight}")

def visualization_tab():
    """Tab untuk visualisasi data"""
    st.markdown("""
    <div class='section-header'>
        <h3>Visualisasi Data dan Hasil Analisis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for existing visualization files
    viz_files = [
        ("Distribusi Target Obesitas", "gambar/01_distribusi_target.png"),
        ("Distribusi Fitur Kategorikal", "gambar/02_distribusi_kategorikal.png"),
        ("Histogram Fitur Numerik", "gambar/03_histogram_numerik.png"),
        ("Deteksi Outlier (Boxplot)", "gambar/04_boxplot_outliers.png"),
        ("Matriks Korelasi", "gambar/05_correlation_matrix.png"),
        ("Distribusi Fitur Biner", "gambar/06_distribusi_biner.png"),
        ("Confusion Matrix Semua Model", "gambar/07_confusion_matrices.png"),
        ("Perbandingan Performa Model", "gambar/08_model_comparison.png"),
        ("Hasil Hyperparameter Tuning", "gambar/09_tuning_comparison.png"),
        ("Confusion Matrix Model Terbaik", "gambar/10_final_confusion_matrix.png")
    ]
    
    # Display existing visualizations
    existing_viz = [(name, path) for name, path in viz_files if os.path.exists(path)]
    
    if existing_viz:
        st.markdown("### Visualisasi dari Analisis Data")
        
        # Create tabs for different visualization categories
        viz_tabs = st.tabs(["EDA", "Model Performance", "Final Results"])
        
        with viz_tabs[0]:
            st.markdown("#### Hasil Exploratory Data Analysis")
            eda_viz = existing_viz[:6]
            
            for i in range(0, len(eda_viz), 2):
                cols = st.columns(2)
                for j, (name, path) in enumerate(eda_viz[i:i+2]):
                    with cols[j]:
                        st.markdown(f"**{name}**")
                        st.image(path, use_column_width=True)
        
        with viz_tabs[1]:
            st.markdown("#### Performa dan Perbandingan Model")
            model_viz = existing_viz[6:9]
            
            for name, path in model_viz:
                st.markdown(f"**{name}**")
                st.image(path, use_column_width=True)
        
        with viz_tabs[2]:
            st.markdown("#### Hasil Akhir Model Terbaik")
            if len(existing_viz) > 9:
                name, path = existing_viz[9]
                st.markdown(f"**{name}**")
                st.image(path, use_column_width=True)
    
    # Generate sample visualizations if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        generate_interactive_visualizations()
    else:
        st.info("Install matplotlib untuk melihat visualisasi interaktif: pip install matplotlib seaborn")
        display_text_analysis()

def generate_interactive_visualizations():
    """Generate visualisasi interaktif"""
    st.markdown("### Visualisasi Interaktif")
    
    # Sample BMI distribution
    st.markdown("#### Distribusi BMI (Data Sampel)")
    np.random.seed(42)
    sample_bmi = np.random.normal(25, 5, 1000)
    sample_bmi = np.clip(sample_bmi, 15, 45)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample_bmi, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi BMI dalam Dataset')
    ax.axvline(x=18.5, color='green', linestyle='--', label='Batas Underweight')
    ax.axvline(x=25, color='orange', linestyle='--', label='Batas Normal')
    ax.axvline(x=30, color='red', linestyle='--', label='Batas Overweight')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance visualization
    st.markdown("#### Tingkat Kepentingan Fitur")
    features = ['BMI', 'Berat Badan', 'Tinggi Badan', 'Usia', 'Riwayat Keluarga', 'Aktivitas Fisik', 'Konsumsi Air']
    importance = [34.2, 29.8, 15.6, 8.9, 6.7, 3.1, 1.7]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, importance, color='lightcoral')
    ax.set_xlabel('Tingkat Kepentingan (%)')
    ax.set_title('Tingkat Kepentingan Fitur dalam Model Random Forest')
    
    for bar, value in zip(bars, importance):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{value}%', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_text_analysis():
    """Display text-based analysis"""
    st.markdown("### Ringkasan Analisis Visual")
    st.markdown("""
    **Temuan Utama dari Visualisasi:**
    
    1. **Distribusi Target**: Dataset memiliki distribusi kelas yang relatif seimbang setelah SMOTE
    2. **Korelasi Fitur**: BMI menunjukkan korelasi tertinggi dengan tingkat obesitas
    3. **Outlier Detection**: Dataset relatif bersih dengan outlier minimal
    4. **Model Performance**: Random Forest konsisten unggul di semua metrik evaluasi
    5. **Feature Importance**: Faktor fisik dominan dalam prediksi obesitas
    """)

# Helper functions untuk prediksi
def prepare_input_data(gender, age, height, weight, family_history, favc, fcvc, ncp, 
                      caec, smoke, ch2o, scc, faf, tue, calc, mtrans, feature_columns):
    """Menyiapkan data input untuk prediksi"""
    try:
        df = pd.DataFrame({
            'Age': [age],
            'Height': [height], 
            'Weight': [weight],
            'family_history_with_overweight': [1 if family_history == 'yes' else 0],
            'FAVC': [1 if favc == 'yes' else 0],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'SMOKE': [1 if smoke == 'yes' else 0],
            'CH2O': [ch2o],
            'SCC': [1 if scc == 'yes' else 0],
            'FAF': [faf],
            'TUE': [tue]
        })
        
        # Tambah BMI
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        
        # Encoding categorical features
        df['Gender_encoded'] = 1 if gender == 'Male' else 0
        
        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        df['CAEC_encoded'] = caec_mapping.get(caec, 0)
        
        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        df['CALC_encoded'] = calc_mapping.get(calc, 0)
        
        mtrans_mapping = {'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public_Transportation': 3, 'Walking': 4}
        df['MTRANS_encoded'] = mtrans_mapping.get(mtrans, 0)
        
        # Filter features sesuai model
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features]
        
    except Exception as e:
        st.error(f"Error menyiapkan data: {str(e)}")
        return None

def make_prediction(model, scaler, input_data):
    """Melakukan prediksi"""
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        return prediction, probabilities
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None

def display_results(prediction, probabilities, label_encoder):
    """Menampilkan hasil prediksi"""
    class_names = label_encoder.classes_
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction] * 100
    
    # Mapping deskripsi dan styling
    class_info = get_class_info()
    info = class_info.get(predicted_class, {
        'name': 'Tidak Diketahui', 'desc': '', 'style': 'result-warning', 
        'advice': 'Konsultasikan dengan tenaga kesehatan'
    })
    
    # Tampilkan hasil utama
    st.markdown(f"""
    <div class='{info["style"]}'>
        <div style='text-align: center;'>
            <h2 style='margin: 0; font-weight: bold;'>Hasil Prediksi Tingkat Obesitas</h2>
            <h3 style='margin: 0.5rem 0; font-weight: bold;'>{info['name']}</h3>
            <p style='font-size: 1.2rem; margin: 0.5rem 0;'><strong>Tingkat Kepercayaan: {confidence:.1f}%</strong></p>
            <p style='margin: 0.5rem 0; font-size: 1.1rem;'>{info['desc']}</p>
        </div>
        <div style='margin-top: 1.5rem; padding: 1rem; background-color: rgba(255,255,255,0.8); border-radius: 5px;'>
            <h5 style='margin: 0 0 0.5rem 0;'>Rekomendasi Kesehatan:</h5>
            <p style='margin: 0; font-weight: 500;'>{info['advice']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribusi probabilitas
    display_probability_distribution(class_names, probabilities, class_info)

def get_class_info():
    """Get class information mapping"""
    return {
        'Insufficient_Weight': {
            'name': 'Berat Badan Kurang',
            'desc': 'Berat badan di bawah rentang normal',
            'style': 'result-warning',
            'advice': 'Pertimbangkan konsultasi dengan ahli gizi untuk menambah berat badan secara sehat'
        },
        'Normal_Weight': {
            'name': 'Berat Badan Normal',
            'desc': 'Berat badan dalam rentang sehat',
            'style': 'result-success',
            'advice': 'Pertahankan pola makan sehat dan aktivitas fisik rutin'
        },
        'Overweight_Level_I': {
            'name': 'Kelebihan Berat Badan Tingkat I',
            'desc': 'Berat badan sedikit di atas normal',
            'style': 'result-warning',
            'advice': 'Tingkatkan aktivitas fisik dan perhatikan pola makan'
        },
        'Overweight_Level_II': {
            'name': 'Kelebihan Berat Badan Tingkat II',
            'desc': 'Kelebihan berat badan tingkat sedang',
            'style': 'result-warning',
            'advice': 'Disarankan konsultasi dengan tenaga kesehatan untuk program penurunan berat badan'
        },
        'Obesity_Type_I': {
            'name': 'Obesitas Tipe I',
            'desc': 'Obesitas kelas I (ringan)',
            'style': 'result-danger',
            'advice': 'Konsultasi dengan dokter untuk program penurunan berat badan yang aman'
        },
        'Obesity_Type_II': {
            'name': 'Obesitas Tipe II',
            'desc': 'Obesitas kelas II (sedang)',
            'style': 'result-danger',
            'advice': 'Perlu penanganan medis profesional untuk manajemen berat badan'
        },
        'Obesity_Type_III': {
            'name': 'Obesitas Tipe III',
            'desc': 'Obesitas kelas III (parah)',
            'style': 'result-danger',
            'advice': 'Segera konsultasi dengan dokter spesialis untuk penanganan komprehensif'
        }
    }

def display_probability_distribution(class_names, probabilities, class_info):
    """Display probability distribution"""
    st.markdown("### Distribusi Probabilitas Semua Kelas")
    
    prob_data = []
    for i, class_name in enumerate(class_names):
        class_display = class_info.get(class_name, {'name': class_name.replace('_', ' ')})['name']
        prob_data.append({
            'Kelas Obesitas': class_display,
            'Probabilitas (%)': f"{probabilities[i] * 100:.1f}%",
            'Nilai Probabilitas': probabilities[i] * 100
        })
    
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values('Nilai Probabilitas', ascending=False)
    
    st.dataframe(prob_df[['Kelas Obesitas', 'Probabilitas (%)']].head(7), use_container_width=True)
    
    # Top 3 predictions
    st.markdown("### Tiga Prediksi Teratas")
    
    cols = st.columns(3)
    top_3 = prob_df.head(3)
    
    for i, col in enumerate(cols):
        if i < len(top_3):
            row = top_3.iloc[i]
            rank = f"Peringkat {i+1}"
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <h5 style='margin: 0;'>{rank}</h5>
                    <p style='margin: 0.5rem 0 0 0; font-weight: bold;'>{row['Kelas Obesitas']}</p>
                    <p style='margin: 0.5rem 0 0 0;'>{row['Probabilitas (%)']}</p>
                </div>
                """, unsafe_allow_html=True)

def display_health_analysis(bmi, age, faf, favc, weight, height, gender):
    """Menampilkan analisis kesehatan lengkap"""
    st.markdown("### Analisis Kesehatan Komprehensif")
    
    col1, col2, col3 = st.columns(3)
    
    # BMI Category
    bmi_info = get_bmi_category(bmi)
    activity_info = get_activity_level(faf)
    diet_info = get_diet_status(favc)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color: {bmi_info["color"]}; margin-top: 0; font-weight: bold;'>Indeks Massa Tubuh</h5>
            <div style='font-size: 2rem; font-weight: bold; color: {bmi_info["color"]};'>{bmi:.1f}</div>
            <p style='margin: 0.5rem 0; font-weight: bold;'><strong>Kategori:</strong> {bmi_info["category"]}</p>
            <p style='margin: 0; font-weight: bold;'><strong>Risiko:</strong> {bmi_info["risk"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color: {activity_info["color"]}; margin-top: 0;'>Tingkat Aktivitas</h5>
            <div style='font-size: 2rem; font-weight: bold; color: {activity_info["color"]};'>{activity_info["level"]}</div>
            <p style='margin: 0.5rem 0;'><strong>Frekuensi:</strong> {faf}/5</p>
            <p style='margin: 0;'><strong>Saran:</strong> {activity_info["advice"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color: {diet_info["color"]}; margin-top: 0;'>Pola Makan</h5>
            <div style='font-size: 1.5rem; font-weight: bold; color: {diet_info["color"]};'>{diet_info["status"]}</div>
            <p style='margin: 0.5rem 0 0 0;'><strong>Saran:</strong> {diet_info["advice"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed recommendations
    display_health_recommendations(bmi, age, faf)
    
    # Health metrics summary table
    display_health_summary(bmi, bmi_info, activity_info, diet_info, faf)

def get_bmi_category(bmi):
    """Get BMI category information"""
    if bmi < 18.5:
        return {"category": "Kurus", "color": "#17a2b8", "risk": "Rendah"}
    elif bmi < 25:
        return {"category": "Normal", "color": "#28a745", "risk": "Optimal"}
    elif bmi < 30:
        return {"category": "Kelebihan Berat Badan", "color": "#ffc107", "risk": "Sedang"}
    else:
        return {"category": "Obesitas", "color": "#dc3545", "risk": "Tinggi"}

def get_activity_level(faf):
    """Get activity level information"""
    if faf <= 2:
        return {"level": "Rendah", "color": "#dc3545", "advice": "Tingkatkan aktivitas fisik"}
    elif faf <= 3:
        return {"level": "Sedang", "color": "#ffc107", "advice": "Pertahankan dan tingkatkan"}
    else:
        return {"level": "Tinggi", "color": "#28a745", "advice": "Sangat baik, pertahankan!"}

def get_diet_status(favc):
    """Get diet status information"""
    if favc == "yes":
        return {"status": "Tinggi Kalori", "color": "#dc3545", "advice": "Kurangi makanan berkalori tinggi"}
    else:
        return {"status": "Seimbang", "color": "#28a745", "advice": "Pertahankan pola makan"}

def display_health_recommendations(bmi, age, faf):
    """Display health recommendations"""
    st.markdown("### Rekomendasi Personal")
    
    recommendations = []
    
    if bmi < 18.5:
        recommendations.extend([
            "**Nutrisi**: Tingkatkan asupan kalori dengan makanan bergizi seimbang",
            "**Olahraga**: Fokus pada latihan kekuatan untuk membangun massa otot"
        ])
    elif bmi >= 30:
        recommendations.extend([
            "**Diet**: Kurangi porsi makan dan pilih makanan rendah kalori",
            "**Kardio**: Lakukan aktivitas kardio minimal 150 menit per minggu"
        ])
    elif bmi >= 25:
        recommendations.extend([
            "**Kalori**: Buat defisit kalori 500-750 kalori per hari",
            "**Aktivitas**: Kombinasikan kardio dan latihan kekuatan"
        ])
    
    if faf <= 2:
        recommendations.extend([
            "**Jadwal**: Mulai dengan olahraga 3x seminggu, durasi 30 menit",
            "**Target**: Tingkatkan secara bertahap hingga 5x seminggu"
        ])
    
    if age >= 40:
        recommendations.extend([
            "**Kesehatan**: Lakukan pemeriksaan kesehatan rutin setiap 6 bulan",
            "**Tulang**: Tambahkan suplemen kalsium dan vitamin D"
        ])
    
    recommendations.extend([
        "**Hidrasi**: Minum air putih minimal 8 gelas per hari",
        "**Tidur**: Pastikan tidur berkualitas 7-9 jam per malam"
    ])
    
    for rec in recommendations[:6]:
        st.markdown(f"- {rec}")

def display_health_summary(bmi, bmi_info, activity_info, diet_info, faf):
    """Display health metrics summary"""
    st.markdown("### Ringkasan Metrik Kesehatan")
    
    health_metrics = {
        'Metrik Kesehatan': [
            'Indeks Massa Tubuh (BMI)',
            'Kategori Berat Badan',
            'Tingkat Aktivitas Fisik',
            'Pola Konsumsi Kalori',
            'Risiko Kesehatan',
            'Rekomendasi Utama'
        ],
        'Nilai/Status': [
            f"{bmi:.1f}",
            bmi_info["category"],
            f"{activity_info['level']} ({faf}/5)",
            diet_info["status"],
            bmi_info["risk"],
            "Konsultasi dengan tenaga kesehatan" if bmi >= 30 or bmi < 18.5 else "Pertahankan gaya hidup sehat"
        ]
    }
    
    health_df = pd.DataFrame(health_metrics)
    st.dataframe(health_df, use_container_width=True)

if __name__ == "__main__":
    main()