import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
from pathlib import Path

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Mango AI | Klasifikasi Kematangan",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_klasifikasi_mangga_knn.joblib"
CSS_PATH = BASE_DIR / "style.css"

if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )

# ==========================================
# STATUS ANALISIS
# ==========================================

if "hasil_analisis" not in st.session_state:
    st.session_state.hasil_analisis = None

if "data_mangga" not in st.session_state:
    st.session_state.data_mangga = None


def ulangi_analisis():
    st.session_state.hasil_analisis = None
    st.session_state.data_mangga = None


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

# ==========================================
# PERSIAPAN DATA
# ==========================================

CATEGORICAL_COLUMNS = ["asal_daerah", "warna_kulit", "musim_panen"]

ALL_COLUMNS = [
    "diameter_cm", "berat_gram", "tebal_kulit_mm", "kadar_gula_brix",
    "asal_daerah", "warna_kulit", "musim_panen"
]


def prepare_input(model, raw_data):
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        expected_features = list(expected_features)

        if all(col in raw_data.columns for col in expected_features):
            return raw_data[expected_features]

        encoded = pd.get_dummies(raw_data, columns=CATEGORICAL_COLUMNS)
        return encoded.reindex(columns=expected_features, fill_value=0)

    n_features = getattr(model, "n_features_in_", None)

    if n_features == len(ALL_COLUMNS):
        return raw_data[ALL_COLUMNS]

    encoded = pd.get_dummies(raw_data, columns=CATEGORICAL_COLUMNS)

    if n_features is not None and len(encoded.columns) == n_features:
        return encoded

    raise ValueError(
        "Format fitur input belum cocok dengan model. "
        "Periksa kembali fitur dan encoding yang digunakan saat pelatihan."
    )


def build_prob_rows(probability_map, top_label):
    """Bangun baris probabilitas sebagai satu blok HTML (tanpa st.columns)."""
    rows = []
    for label, prob in sorted(
        probability_map.items(), key=lambda item: item[1], reverse=True
    ):
        pct = max(0.0, min(100.0, prob * 100))
        top_class = " top" if label == top_label else ""
        badge = '<em class="prob-badge">Prediksi</em>' if label == top_label else ""
        rows.append(
            f'<div class="prob-row{top_class}">'
            f'<div class="prob-head">'
            f'<span class="prob-name">{label}{badge}</span>'
            f'<span class="prob-pct">{pct:.1f}%</span>'
            f'</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{pct:.2f}%"></div>'
            f'</div>'
            f'</div>'
        )
    return '<div class="prob-card">' + "".join(rows) + "</div>"


# ==========================================
# HEADER & HERO
# ==========================================

st.markdown(
    """
<div class="topbar">
<div class="brand">
<div class="brand-icon">🥭</div>
<div>
<div class="brand-name">MANGO<span>AI</span></div>
<div class="brand-caption">SMART FRUIT CLASSIFICATION</div>
</div>
</div>
<div class="status-pill"><span class="status-dot"></span>MACHINE LEARNING</div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero">
<div class="hero-copy">
<div class="eyebrow">✦ INTELLIGENT MANGO ANALYSIS</div>
<h1>Kenali Kematangan<br>Mangga dengan AI.</h1>
<p>Analisis karakteristik mangga melalui kadar gula, diameter,
ketebalan kulit, dan berat untuk mendapatkan prediksi kelas
menggunakan algoritma K-Nearest Neighbors.</p>
<div class="hero-tags">
<span>◆ Algoritma KNN</span>
<span>◉ Prediksi Otomatis</span>
<span>✦ Analisis Karakteristik</span>
</div>
</div>
<div class="hero-visual">
<div class="ring ring-1"></div>
<div class="ring ring-2"></div>
<div class="ring ring-3"></div>
<div class="ring ring-4"></div>
<div class="fruit-emoji">🥭</div>
<div class="floating-card card-top">
<div class="fc-icon">✦</div>
<div><div class="fc-title">AI Powered</div><div class="fc-sub">Smart Prediction</div></div>
</div>
<div class="floating-card card-bottom">
<span class="mini-dot"></span>
<div><div class="fc-title">4 Parameter</div><div class="fc-sub">Fruit Characteristics</div></div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True
)

if model_error:
    st.error(
        f"Model gagal dimuat: {model_error}\n\n"
        "Pastikan file model berada satu folder dengan app.py."
    )
    st.stop()

# ==========================================
# FORM INPUT
# ==========================================

st.markdown(
    """
<div class="section-heading">
<div>
<div class="eyebrow">INPUT PARAMETERS</div>
<h2>Karakteristik Mangga</h2>
<p>Masukkan karakteristik buah yang ingin dianalisis.</p>
</div>
<div class="step-label">STEP 01 / 02</div>
</div>
""",
    unsafe_allow_html=True
)

with st.form("mango_form"):

    st.markdown(
        '<div class="input-section-title">01 &nbsp; Ukuran dan Komposisi</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        diameter = st.slider(
            "Diameter buah (cm)", min_value=4.0, max_value=12.0,
            value=7.0, step=0.1,
            help="Perkiraan diameter mangga dalam sentimeter."
        )
        berat = st.slider(
            "Berat buah (gram)", min_value=50, max_value=500,
            value=200, step=5,
            help="Berat satu buah mangga dalam gram."
        )

    with col2:
        tebal_kulit = st.slider(
            "Ketebalan kulit (mm)", min_value=0.1, max_value=2.0,
            value=0.5, step=0.05,
            help="Perkiraan ketebalan kulit mangga dalam milimeter."
        )
        kadar_gula = st.slider(
            "Kadar gula (°Brix)", min_value=4.0, max_value=25.0,
            value=12.0, step=0.5,
            help="Perkiraan kadar gula berdasarkan skala Brix."
        )

    st.markdown(
        '<div class="input-section-title second-title">02 &nbsp; Identitas dan Kondisi Buah</div>',
        unsafe_allow_html=True
    )

    col3, col4, col5 = st.columns(3, gap="medium")

    with col3:
        asal_daerah = st.selectbox(
            "Asal daerah",
            options=["Probolinggo", "Indramayu", "Kediri", "Lainnya"]
        )

    with col4:
        warna_kulit = st.selectbox(
            "Warna kulit",
            options=["hijau", "hijau kekuningan", "kuning kehijauan",
                     "kuning", "kuning kecokelatan"]
        )

    with col5:
        musim_panen = st.selectbox("Musim panen", options=["hujan", "kemarau"])

    st.caption(
        "Pastikan nilai yang dipilih sesuai dengan karakteristik "
        "mangga yang akan diperiksa."
    )

    submitted = st.form_submit_button(
        "🥭  ANALISIS KEMATANGAN MANGGA",
        use_container_width=True
    )

# ==========================================
# PROSES PREDIKSI
# ==========================================

if submitted:
    raw_input = pd.DataFrame([{
        "diameter_cm": diameter,
        "berat_gram": berat,
        "tebal_kulit_mm": tebal_kulit,
        "kadar_gula_brix": kadar_gula,
        "asal_daerah": asal_daerah,
        "warna_kulit": warna_kulit,
        "musim_panen": musim_panen
    }])

    try:
        model_input = prepare_input(model, raw_input)
        prediction = model.predict(model_input)[0]
        prediction_text = str(prediction).strip()
        prediction_lower = prediction_text.lower()

        if "terlalu" in prediction_lower:
            result_icon, result_class = "🍂", "overripe"
            result_description = (
                "Mangga diprediksi berada pada kondisi terlalu matang. "
                "Perhatikan tekstur dan kondisi buah sebelum dikonsumsi."
            )
        elif "mentah" in prediction_lower:
            result_icon, result_class = "🌿", "unripe"
            result_description = (
                "Mangga diprediksi masih mentah dan mungkin memerlukan "
                "waktu pematangan lebih lanjut."
            )
        elif "matang" in prediction_lower:
            result_icon, result_class = "🥭", "ripe"
            result_description = "Mangga diprediksi berada pada tingkat kematangan matang."
        else:
            result_icon, result_class = "🥭", "general"
            result_description = "Model berhasil menghasilkan kelas prediksi untuk mangga ini."

        probability_map = {}
        confidence_pct = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            probability_map = {
                str(label): float(prob)
                for label, prob in zip(list(model.classes_), probabilities)
            }
            confidence = probability_map.get(prediction_text)
            if confidence is not None:
                confidence_pct = confidence * 100

        st.session_state.hasil_analisis = {
            "prediction_text": prediction_text,
            "result_icon": result_icon,
            "result_class": result_class,
            "result_description": result_description,
            "probability_map": probability_map,
            "confidence_pct": confidence_pct
        }

        st.session_state.data_mangga = {
            "diameter": diameter,
            "berat": berat,
            "tebal_kulit": tebal_kulit,
            "kadar_gula": kadar_gula,
            "asal_daerah": asal_daerah,
            "warna_kulit": warna_kulit,
            "musim_panen": musim_panen
        }

    except Exception as e:
        st.error(
            "Prediksi belum dapat dilakukan karena format input tidak "
            "sesuai dengan model yang tersimpan."
        )
        st.code(str(e))
        st.warning(
            "Pastikan model dilatih menggunakan fitur diameter_cm, "
            "berat_gram, tebal_kulit_mm, kadar_gula_brix, asal_daerah, "
            "warna_kulit, dan musim_panen dengan encoding yang sesuai."
        )

# ==========================================
# HASIL
# ==========================================

if st.session_state.hasil_analisis is not None:

    hasil = st.session_state.hasil_analisis
    data = st.session_state.data_mangga

    st.markdown('<div id="hasil-prediksi"></div>', unsafe_allow_html=True)

    components.html(
        """
        <script>
        setTimeout(function() {
            const target = window.parent.document.getElementById("hasil-prediksi");
            if (target) {
                target.scrollIntoView({behavior: "smooth", block: "start"});
            }
        }, 350);
        </script>
        """,
        height=0
    )

    st.markdown(
        """
<div class="section-heading result-heading">
<div>
<div class="eyebrow">ANALYSIS COMPLETE</div>
<h2>Hasil Prediksi</h2>
<p>Berikut hasil analisis berdasarkan data yang dimasukkan.</p>
</div>
<div class="step-label">STEP 02 / 02</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="result-card {hasil['result_class']}">
<div class="result-left">
<div class="result-icon">{hasil['result_icon']}</div>
<div class="result-label">HASIL KLASIFIKASI</div>
<div class="result-name">{hasil['prediction_text']}</div>
<p>{hasil['result_description']}</p>
</div>
<div class="result-right">
<div class="result-ring">
<div class="result-ring-inner">
<span class="ring-symbol">✦</span>
<span class="ring-caption">MANGO AI</span>
</div>
</div>
<div class="result-right-caption">MODEL KNN</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.button(
        "↻ Analisis Mangga Lain",
        key="ulang_dekat_hasil",
        use_container_width=True,
        on_click=ulangi_analisis
    )

    # ---------- Keyakinan model ----------

    st.markdown(
        '<div class="subsection-title">Tingkat Keyakinan Model</div>',
        unsafe_allow_html=True
    )

    probability_map = hasil["probability_map"]
    confidence_pct = hasil["confidence_pct"]

    if probability_map:

        if confidence_pct is not None:
            conf_col1, conf_col2 = st.columns([2, 1], gap="large")

            with conf_col1:
                st.markdown(
                    f"""
<div class="confidence-card">
<div class="confidence-top">
<span>Keyakinan pada hasil prediksi</span>
<strong>{confidence_pct:.1f}%</strong>
</div>
<div class="confidence-track">
<div class="confidence-fill" style="width:{confidence_pct:.2f}%"></div>
</div>
<div class="confidence-foot">Nilai probabilitas yang dihasilkan model KNN.</div>
</div>
""",
                    unsafe_allow_html=True
                )

            with conf_col2:
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-icon">✦</div>
<div class="metric-label">PREDIKSI UTAMA</div>
<div class="metric-value">{confidence_pct:.1f}%</div>
<div class="metric-desc">Probabilitas kelas terpilih</div>
</div>
""",
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="subsection-title small-subsection">Probabilitas Setiap Kelas</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            build_prob_rows(probability_map, hasil["prediction_text"]),
            unsafe_allow_html=True
        )

        st.caption(
            "Catatan: probabilitas model bukan jaminan mutlak bahwa "
            "prediksi benar. Hasil dipengaruhi kualitas data pelatihan."
        )

    else:
        st.info(
            "Model tidak menyediakan predict_proba, sehingga persentase "
            "keyakinan tidak dapat ditampilkan secara langsung."
        )

    # ---------- Ringkasan input ----------

    st.markdown(
        '<div class="subsection-title">Ringkasan Data yang Dianalisis</div>',
        unsafe_allow_html=True
    )

    summary_cols = st.columns(4, gap="medium")

    summary_items = [
        ("DIAMETER", f"{data['diameter']:.1f} cm"),
        ("BERAT", f"{data['berat']} gram"),
        ("KETEBALAN KULIT", f"{data['tebal_kulit']:.2f} mm"),
        ("KADAR GULA", f"{data['kadar_gula']:.1f} °Brix")
    ]

    for column, (label, value) in zip(summary_cols, summary_items):
        with column:
            st.markdown(
                f"""
<div class="summary-card">
<div class="summary-label">{label}</div>
<div class="summary-value">{value}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown(
        f"""
<div class="detail-card">
<div><span>Asal daerah</span><strong>{data['asal_daerah']}</strong></div>
<div><span>Warna kulit</span><strong>{data['warna_kulit'].title()}</strong></div>
<div><span>Musim panen</span><strong>{data['musim_panen'].title()}</strong></div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="disclaimer">
<strong>Informasi:</strong> Hasil ini merupakan prediksi berdasarkan model
machine learning dan bukan pemeriksaan fisik langsung terhadap buah.
</div>
""",
        unsafe_allow_html=True
    )

    st.button(
        "↻ Analisis Mangga Lain",
        key="ulang_bawah",
        use_container_width=True,
        on_click=ulangi_analisis
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
<div class="footer">
<div class="footer-brand">🥭 MANGO AI</div>
<div>SMART CLASSIFICATION SYSTEM</div>
<div class="footer-note">Aplikasi klasifikasi kematangan mangga berbasis Machine Learning</div>
<div class="footer-note">Author : Alya Andina</div>
</div>
""",
    unsafe_allow_html=True
)