import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
from PIL import Image
import docx
import PyPDF2
import io

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Heart Disease AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg,#050816,#0b1120,#111827);
    color: white;
    overflow-x: hidden;
}

/* Animated Background */
.stApp::before {
    content: "";
    position: fixed;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(255,0,120,0.15), transparent 70%);
    top: -250px;
    left: -250px;
    animation: pulse 8s infinite alternate;
    z-index: -1;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(0,200,255,0.15), transparent 70%);
    bottom: -250px;
    right: -250px;
    animation: pulse2 10s infinite alternate;
    z-index: -1;
}

@keyframes pulse {
    from {transform: scale(1);}
    to {transform: scale(1.3);}
}

@keyframes pulse2 {
    from {transform: scale(1);}
    to {transform: scale(1.4);}
}

/* Title */
.main-title {
    font-size: 65px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(to right,#ff4b91,#00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-10px);}
    100% {transform: translateY(0px);}
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 22px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: 0.4s;
}

.card:hover {
    transform: translateY(-10px);
    box-shadow: 0 12px 40px rgba(0,255,255,0.3);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10,10,20,0.95);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#ff006e,#00d4ff);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 12px 30px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px #00d4ff;
}

/* Prediction Text */
.prediction {
    font-size: 24px;
    font-weight: bold;
    color: #00ffcc;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #

lr_model = joblib.load("models/logistic_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
ann_model = joblib.load("models/ann_model.pkl")

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="main-title">❤️ AI Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced Machine Learning & Neural Network Healthcare Dashboard</div>',
    unsafe_allow_html=True
)

# ---------------- HERO SECTION ---------------- #

st.markdown("""
<div style='text-align:center; margin-top:20px; margin-bottom:20px;'>

<h2 style='color:white;'>🏥 AI Powered Smart Healthcare System</h2>

<p style='color:#cbd5e1; font-size:18px;'>
Predict heart disease risk using advanced AI algorithms.
</p>

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d",
        use_container_width=True
    )

with col2:
    st.image(
        "https://images.unsplash.com/photo-1584515933487-779824d29309",
        use_container_width=True
    )

with col3:
    st.image(
        "https://images.unsplash.com/photo-1537368910025-700350fe46c7",
        use_container_width=True
    )

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload CSV / PDF / DOCX / IMAGE",
    type=["csv", "pdf", "docx", "png", "jpg", "jpeg"]
)

# ---------------- CSV FILE ---------------- #

if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1]

    # CSV
    if file_type == "csv":

        df = pd.read_csv(uploaded_file)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("🧹 Missing Values")
        st.write(df.isnull().sum())

        st.subheader("📈 Summary Statistics")
        st.write(df.describe())

        st.markdown('</div>', unsafe_allow_html=True)

        # Heatmap

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🔥 Correlation Heatmap")

        corr = df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # PDF
    elif file_type == "pdf":

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📄 PDF Preview")

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        st.text_area("PDF Content", text, height=300)

        st.success("✅ PDF Uploaded Successfully")

        st.markdown('</div>', unsafe_allow_html=True)

    # DOCX
    elif file_type == "docx":

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📝 DOCX Preview")

        doc = docx.Document(uploaded_file)

        full_text = []

        for para in doc.paragraphs:
            full_text.append(para.text)

        st.text_area(
            "Document Content",
            "\n".join(full_text),
            height=300
        )

        st.success("✅ DOCX Uploaded Successfully")

        st.markdown('</div>', unsafe_allow_html=True)

    # IMAGE
    elif file_type in ["png", "jpg", "jpeg"]:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🖼 Uploaded Medical Image")

        image = Image.open(uploaded_file)

        st.image(image, use_container_width=True)

        st.success(
            "✅ Image Uploaded Successfully"
        )

        st.warning(
            "⚠ Image-based heart disease prediction requires deep learning medical imaging models."
        )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.markdown("""
<h1 style='text-align:center; color:#00d4ff;'>
🩺 Patient Information
</h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

age = st.sidebar.slider("👴 Age", 20, 100, 50)

sex = st.sidebar.selectbox(
    "⚧ Sex",
    options=[0,1],
    format_func=lambda x:
    "0 = Male 👨" if x == 0 else "1 = Female 👩"
)

cp = st.sidebar.selectbox(
    "💓 Chest Pain Type",
    options=[0,1,2,3],
    format_func=lambda x: {
        0:"0 = Typical Angina",
        1:"1 = Atypical Angina",
        2:"2 = Non-anginal Pain",
        3:"3 = Asymptomatic"
    }[x]
)

trestbps = st.sidebar.slider(
    "🩸 Resting Blood Pressure",
    80,
    200,
    120
)

chol = st.sidebar.slider(
    "🥗 Cholesterol",
    100,
    600,
    200
)

fbs = st.sidebar.selectbox(
    "🍬 Fasting Blood Sugar",
    options=[0,1],
    format_func=lambda x:
    "0 = Below 120 mg/dl" if x == 0
    else "1 = Above 120 mg/dl"
)

restecg = st.sidebar.selectbox(
    "📈 Rest ECG",
    options=[0,1,2],
    format_func=lambda x: {
        0:"0 = Normal",
        1:"1 = ST-T Abnormality",
        2:"2 = Left Ventricular Hypertrophy"
    }[x]
)

thalach = st.sidebar.slider(
    "❤️ Maximum Heart Rate",
    60,
    220,
    150
)

exang = st.sidebar.selectbox(
    "🏃 Exercise Angina",
    options=[0,1],
    format_func=lambda x:
    "0 = No" if x == 0 else "1 = Yes"
)

oldpeak = st.sidebar.slider(
    "📉 Oldpeak",
    0.0,
    6.0,
    1.0
)

slope = st.sidebar.selectbox(
    "📊 Slope",
    options=[0,1,2],
    format_func=lambda x: {
        0:"0 = Upsloping",
        1:"1 = Flat",
        2:"2 = Downsloping"
    }[x]
)

ca = st.sidebar.selectbox(
    "🧬 Number of Major Vessels",
    options=[0,1,2,3,4]
)

thal = st.sidebar.selectbox(
    "🩺 Thalassemia",
    options=[0,1,2,3],
    format_func=lambda x: {
        0:"0 = Normal",
        1:"1 = Fixed Defect",
        2:"2 = Reversible Defect",
        3:"3 = Unknown"
    }[x]
)

st.sidebar.markdown("---")

# ---------------- PREDICTION ---------------- #

if st.sidebar.button("🚀 Predict Heart Disease"):

    input_data = np.array([[

        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]])

    input_scaled = scaler.transform(input_data)

    lr_pred = lr_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    svm_pred = svm_model.predict(input_scaled)[0]
    ann_pred = ann_model.predict(input_scaled)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧠 AI Prediction Results")

    st.markdown(
        f'<div class="prediction">Logistic Regression: {"❤️ Disease" if lr_pred else "✅ No Disease"}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="prediction">Random Forest: {"❤️ Disease" if rf_pred else "✅ No Disease"}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="prediction">SVM: {"❤️ Disease" if svm_pred else "✅ No Disease"}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="prediction">ANN: {"❤️ Disease" if ann_pred else "✅ No Disease"}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- RISK METER ---------------- #

    risk_value = int(
        (lr_pred + rf_pred + svm_pred + ann_pred) / 4 * 100
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        title={'text': "Heart Disease Risk %"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0,40], 'color': "green"},
                {'range': [40,70], 'color': "orange"},
                {'range': [70,100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ---------------- DOWNLOAD REPORT ---------------- #

    report = f"""
HEART DISEASE AI REPORT

Logistic Regression:
{"Disease" if lr_pred else "No Disease"}

Random Forest:
{"Disease" if rf_pred else "No Disease"}

SVM:
{"Disease" if svm_pred else "No Disease"}

ANN:
{"Disease" if ann_pred else "No Disease"}

Risk Score:
{risk_value}%
"""

    st.download_button(
        "⬇ Download Report",
        report,
        file_name="heart_report.txt"
    )

    # ---------------- WHATSAPP ---------------- #

    encoded_message = urllib.parse.quote(report)

    whatsapp_url = f"https://wa.me/?text={encoded_message}"

    st.markdown(
        f"""
        <a href="{whatsapp_url}" target="_blank">
            <button style="
                background:green;
                color:white;
                padding:15px;
                border:none;
                border-radius:10px;
                font-size:18px;">
            📲 Share on WhatsApp
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

# ---------------- AI CHATBOT ---------------- #

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h2 style='color:white;'>🤖 AI Health Assistant</h2>
<p style='color:#cbd5e1;'>
Ask questions about heart health.
</p>
</div>
""", unsafe_allow_html=True)

user_question = st.text_input(
    "💬 Ask AI Health Assistant"
)

if user_question:

    if "heart" in user_question.lower():

        st.success(
            "❤️ Regular exercise and healthy food help maintain heart health."
        )

    elif "cholesterol" in user_question.lower():

        st.success(
            "🥗 Reduce oily food and exercise daily."
        )

    elif "blood pressure" in user_question.lower():

        st.success(
            "🩺 High blood pressure increases heart disease risk."
        )

    elif "exercise" in user_question.lower():

        st.success(
            "🏃 Daily exercise improves cardiovascular health."
        )

    else:

        st.success(
            "🤖 Please consult a medical professional."
        )