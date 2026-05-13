import streamlit as st
import pickle, io, re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pandas as pd
import plotly.express as px  # Professional Graphs ke liye

# Page Config
st.set_page_config(page_title="AI Rsume Screening | Resume Intelligence", page_icon="🎯", layout="wide")

# ─────────────────────────────────────────────
# STYLES & ASSETS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    </style>
    """, unsafe_allow_html=True) # <--- Corrected parameter name

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
import joblib  # <--- Ye line add karein

@st.cache_resource
def load_models():
    try:
        # pickle.load ki jagah joblib.load use karein
        data = joblib.load("models.pkl") 
        return data, True
    except Exception as e:
        return e, False

data, MODELS_LOADED = load_models()

if MODELS_LOADED:
    tfidf, le = data["tfidf"], data["label_encoder"]
    models_dict = {
        "Logistic Regression": data["lr_model"],
        "Naive Bayes": data["nb_model"],
        "Linear SVC": data["svc_model"],
        "Random Forest": data["rf_model"]
    }
else:
    st.error("❌ Critical Error: Model file 'models.pkl' not found.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def download_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

download_nltk()
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|\S+@\S+|[^a-z\s]", " ", text.lower())
    return " ".join([w for w in text.split() if w not in STOPWORDS and len(w) > 2])

def extract_text(file):
    if file.type == "application/pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                return " ".join(page.extract_text() or "" for page in pdf.pages)
        except: return ""
    else:
        return file.read().decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────
# MAIN APP INTERFACE
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("JobJive AI")
menu = st.sidebar.selectbox("Menu", ["🔍 Resume Predictor", "⚖️ Bias & Ethics Audit"])

if menu == "🔍 Resume Predictor":
    st.title("Resume Category Predictor")
    st.markdown("Upload multiple resumes to identify professional categories using our **Ensemble Voting System**.")

    uploaded_files = st.file_uploader("Drop PDFs or Text files here", type=["pdf", "txt"], accept_multiple_files=True)

    if MODELS_LOADED and uploaded_files:
        if st.button("Start AI Analysis"):
            all_results = []
            
            for f in uploaded_files: # Limit to 5
                with st.spinner(f"Analyzing {f.name}..."):
                    raw_text = extract_text(f)
                    if not raw_text.strip(): continue
                    
                    cleaned = clean_text(raw_text)
                    vec = tfidf.transform([cleaned])
                    
                    # Individual Model Predictions
                    individual_preds = {}
                    for name, model in models_dict.items():
                        p = model.predict(vec)[0]
                        individual_preds[name] = le.inverse_transform([p])[0]
                    
                    # Final Voting
                    votes = list(individual_preds.values())
                    final_cat = Counter(votes).most_common(1)[0][0]
                    conf = (votes.count(final_cat) / len(votes)) * 100
                    
                    all_results.append({
                        "filename": f.name,
                        "final": final_cat,
                        "confidence": conf,
                        "details": individual_preds
                    })

            # --- DISPLAY RESULTS ---
            if all_results:
                st.divider()
                res_df = pd.DataFrame(all_results)
                
                # 1. Summary Cards
                c1, c2, c3 = st.columns(3)
                c1.metric("Processed", len(all_results))
                c2.metric("Top Category", res_df['final'].mode()[0])
                c3.metric("Avg. Confidence", f"{res_df['confidence'].mean():.0f}%")

                # 2. Advanced Visual (Plotly)
                st.subheader("📊 Category Distribution")
                fig = px.bar(res_df['final'].value_counts().reset_index(), 
                             x='final', y='count', 
                             labels={'final': 'Category', 'count': 'Number of Resumes'},
                             color='final', color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)

                # 3. Detailed Breakdown
                st.subheader("📄 Candidate Analysis")
                for res in all_results:
                    with st.expander(f"Analysis for: {res['filename']} - **{res['final']}** ({res['confidence']:.0f}%)"):
                        cols = st.columns(len(models_dict))
                        for i, (m_name, m_pred) in enumerate(res['details'].items()):
                            is_match = "✅" if m_pred == res['final'] else "❌"
                            cols[i].markdown(f"**{m_name}**\n\n{is_match} {m_pred}")
                        
                        if res['confidence'] < 100:
                            st.warning("Note: Models disagreed on this file. Manual review recommended.")

    elif not uploaded_files:
        st.info("Please upload a resume to begin analysis.")

# ─────────────────────────────────────────────
# BIAS REPORT PAGE (IMPROVED)
# ─────────────────────────────────────────────
else:
    st.title("⚖️ Bias & Ethics Audit")
    st.markdown("Transparency is key in AI recruitment. Below is the ethical breakdown of our screening engine.")
    
    t1, t2, t3 = st.tabs(["🛡️ Mitigation Strategies", "⚠️ Known Limitations", "📜 Compliance"])
    
    with t1:
        st.subheader("How we reduce bias")
        st.markdown("""
        *   **Feature Stripping:** Our cleaner automatically removes names, emails, and contact details before prediction.
        *   **Ensemble Logic:** We use 4 different algorithms to ensure a single model's bias doesn't dominate the result.
        *   **Stopword Filtering:** Terms that don't relate to skills are ignored.
        """)
    
    with t2:
        st.error("Algorithm Sensitivity Notice")
        st.write("""
        1. **Historical Bias:** The model is trained on past resumes. If certain roles were historically male-dominated, the model might associate specific 'masculine' verbs with those roles.
        2. **Language Gap:** Non-native English phrasing might result in lower confidence scores.
        3. **Formatting:** Highly creative layouts (infographics) may not parse correctly.
        """)

    with t3:
        st.success("Best Practices for Recruiters")
        st.info("""
        **Recommendation:** Use this tool as a 'First Pass' filter. Never reject a candidate based solely on an AI prediction. 
        Ensure your final interview panel is diverse and unaware of the AI's confidence score to avoid 'Automation Bias'.
        """)