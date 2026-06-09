import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from imblearn.over_sampling import SMOTE

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp { background-color: #0f1117; color: white; }
section[data-testid="stSidebar"] { background-color: #111827; }
section[data-testid="stSidebar"] * { color: white !important; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.main-title { font-size: 50px; font-weight: bold; text-align: center; color: white; }
.section-title { color: white; font-size: 38px; font-weight: bold; margin-bottom: 20px; }
.metric-card { background: #1f2937; padding: 25px; border-radius: 18px; text-align: center; border: 1px solid #374151; }
.metric-value { color: #22c55e; font-size: 36px; font-weight: bold; }
.metric-label { color: #d1d5db; font-size: 16px; }
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #15803d, #22c55e);
    color: white; border: none; border-radius: 12px;
    padding: 14px; font-size: 18px; font-weight: bold;
}
.stButton>button:hover { background: linear-gradient(90deg, #166534, #16a34a); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown('<div class="sidebar-title">Sentiment AI</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Online Learning Sentiment Analysis</div>', unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border:1px solid #d1d5db;'>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🧹 Text Preprocessing",
        "📈 Test Dataset Evaluation",
        "🎓 Sentiment Demo"
    ]
)

st.sidebar.markdown("<hr style='border:1px solid #d1d5db;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="footer-text">
<b>Kelompok 9 - LC01</b><br><br>
1. Samuel Christoff<br>
2. Jovin Prasetia Willim<br>
3. Albani Kalam Haq
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border:1px solid #d1d5db;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="footer-text" style="text-align:center;">
Machine Learning Project<br>Binus University
</div>
""", unsafe_allow_html=True)

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text)
    text = text.replace('"', '').replace("'", "")
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================================================
# TEST CSV PARSER
# FIX: test.csv has a malformed quoting structure where the label
#      is embedded inside the quoted text field. Parse it manually.
# =========================================================

@st.cache_data(show_spinner="Loading test data…")
def load_test_csv(path="test.csv"):
    """
    test.csv rows look like:
        "\"text content here\", 0"
    The label is embedded at the end of the quoted string.
    This function extracts text and label correctly.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    lines = raw.splitlines()
    records = []
    for line in lines[1:]:   # skip header
        line = line.strip()
        if not line:
            continue
        # strip outer wrapping quote
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        # extract trailing label digit
        m = re.match(r'^(.*),\s*([01])\s*$', line)
        if m:
            text = m.group(1).strip().strip('"')
            label = int(m.group(2))
            records.append({"text": text, "label": label})

    return pd.DataFrame(records)

# =========================================================
# MODEL TRAINING WITH SMOTE
# FIX 1: use correct column 'review_text' (not 'full_text')
# FIX 2: use the real 'sentiment' column; map 3 classes → binary
#         (0=Negative → 0, 1=Neutral → 0, 2=Positive → 1)
# FIX 3: keep SVM as LinearSVC for speed & predict_proba-like support
# =========================================================

@st.cache_resource(show_spinner="Training balanced model…")
def train_balanced_model():
    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip().str.lower()

    # FIX 1 & 2: correct column + use real labels mapped to binary
    df["cleaned"] = df["review_text"].apply(clean_text)
    df["label"]   = df["sentiment"].map({0: 0, 1: 0, 2: 1}).fillna(0).astype(int)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(df["cleaned"])
    y = df["label"]

    # SMOTE — guard k_neighbors against tiny minority class
    min_samples = int(y.value_counts().min())
    k = min(5, min_samples - 1)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)

    model_lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=42
    )
    model_lr.fit(X_res, y_res)

    model_nb = MultinomialNB()
    model_nb.fit(X_res, y_res)

    model_svm = LinearSVC(class_weight="balanced", max_iter=1000, random_state=42)
    model_svm.fit(X_res, y_res)

    label_dist = df["label"].value_counts().to_dict()
    return model_lr, model_nb, model_svm, vec, df, label_dist, X_res, y_res

model_lr, model_nb, model_svm, vectorizer, train_df, label_dist, X_res, y_res = train_balanced_model()

# =========================================================
# HOME
# =========================================================

if menu == "🏠 Home":
    st.markdown('<br><div class="main-title">🎓 Sentiment Analysis of Online Learning</div>', unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image("https://images.unsplash.com/photo-1522202176988-66273c2fd55f", width=800)

    st.write("")
    st.markdown("""
    <div style='background-color:#111827;padding:20px;border-radius:15px;color:white;'>
    <h3>📌 About This Project</h3>
    This project analyzes student opinions regarding online learning using
    Natural Language Processing (NLP) and Machine Learning techniques.<br><br>
    The system classifies opinions into <b>Positive</b> and <b>Negative</b> sentiments
    and compares the performance of several classification algorithms:
    <ul>
        <li>Naive Bayes</li>
        <li>Logistic Regression</li>
        <li>Support Vector Machine (SVM)</li>
    </ul>
    </div>

    <br>

    <div style='background-color:#1a3a2a;padding:20px;border-radius:15px;color:white;border:1px solid #22c55e;'>
    <h3>✅ Bugs Fixed</h3>
    <ul>
        <li><b>Wrong column name:</b> <code>full_text</code> → <code>review_text</code> (train.csv has no full_text column)</li>
        <li><b>Ignored real labels:</b> Now uses the actual <code>sentiment</code> column (0/1/2) mapped to binary instead of keyword guessing</li>
        <li><b>Broken test.csv parser:</b> Custom parser now correctly extracts text and label from malformed CSV quoting</li>
        <li><b>No evaluation metrics:</b> Test Dataset page now shows accuracy, classification report, and confusion matrix</li>
        <li><b>Slow confidence loop:</b> <code>predict_proba</code> is now called once on the full matrix, not per-row</li>
    </ul>
    </div>

    <br>

    <div style='background-color:#111827;padding:20px;border-radius:15px;color:white;'>
    <h3>👥 Project Team</h3>
    Group 9 - LC01<br><br>
    1. Samuel Christoff<br>
    2. Jovin Prasetia Willim<br>
    3. Albani Kalam Haq
    </div>
    <br>
    """, unsafe_allow_html=True)

# =========================================================
# DATASET OVERVIEW
# =========================================================

elif menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip().str.lower()
    st.subheader("Dataset Information")
    st.write(f"Total Rows: {len(df)}")
    st.write(f"Total Columns: {len(df.columns)}")

    st.subheader("Sentiment Distribution (Binary Labels)")
    col1, col2 = st.columns(2)
    neg_count = label_dist.get(0, 0)
    pos_count = label_dist.get(1, 0)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#ef4444;">{neg_count}</div>
            <div class="metric-label">Negative Samples (sentiment 0 & 1)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pos_count}</div>
            <div class="metric-label">Positive Samples (sentiment 2)</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div style='background-color:#1a2a3a;padding:15px;border-radius:10px;color:#93c5fd;'>
    ℹ️ Original <code>sentiment</code> column has 3 classes: 0 (Negative), 1 (Neutral), 2 (Positive).
    They are mapped to binary: 0 &amp; 1 → Negative, 2 → Positive.
    After SMOTE oversampling, both classes have equal representation during training.
    </div>""", unsafe_allow_html=True)

    st.write("")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#1f2937")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    axes[0].bar(["Negative", "Positive"], [neg_count, pos_count], color=["#ef4444", "#22c55e"])
    axes[0].set_title("Before SMOTE (Original)", color="white")
    axes[0].set_ylabel("Count", color="white")

    unique, counts = np.unique(y_res, return_counts=True)
    after = dict(zip(unique.tolist(), counts.tolist()))
    axes[1].bar(["Negative", "Positive"], [after.get(0, 0), after.get(1, 0)], color=["#ef4444", "#22c55e"])
    axes[1].set_title("After SMOTE (Balanced)", color="white")
    axes[1].set_ylabel("Count", color="white")

    st.pyplot(fig)

    st.subheader("Sample Training Data")
    # FIX: correct column name review_text
    display_df = train_df[["review_text", "cleaned", "label"]].copy()
    display_df["label"] = display_df["label"].map({0: "Negative", 1: "Positive"})
    st.dataframe(display_df.head(20), use_container_width=True)

# =========================================================
# TEXT PREPROCESSING
# =========================================================

elif menu == "🧹 Text Preprocessing":
    st.title("🧹 Text Preprocessing")

    df = pd.read_csv("train.csv")
    df.columns = df.columns.str.strip().str.lower()
    # FIX: correct column name
    sample = df["review_text"].dropna().head(20)
    preprocess_df = pd.DataFrame({
        "Original Text": sample.values,
        "Cleaned Text": sample.apply(clean_text).values
    })

    st.dataframe(preprocess_df, use_container_width=True)

    idx = st.slider("Select Example", 0, len(preprocess_df) - 1, 0)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.write(preprocess_df.iloc[idx]["Original Text"])
    with col2:
        st.subheader("After")
        st.write(preprocess_df.iloc[idx]["Cleaned Text"])

# =========================================================
# TEST DATASET EVALUATION
# FIX: use custom parser for malformed test.csv
# FIX: actually evaluate against true labels (accuracy, report, confusion matrix)
# FIX: compute all confidences in one vectorized call
# =========================================================

elif menu == "📈 Test Dataset Evaluation":
    st.title("📈 Test Dataset Evaluation")

    test_df = load_test_csv("test.csv")
    test_df["cleaned"] = test_df["text"].apply(clean_text)
    X_test = vectorizer.transform(test_df["cleaned"])
    y_true = test_df["label"].values

    st.subheader("Model Comparison on Test Set")

    models = {
        "Logistic Regression (SMOTE)": model_lr,
        "Naive Bayes (SMOTE)":         model_nb,
        "SVM (SMOTE)":                 model_svm,
    }

    for model_name, m in models.items():
        preds = m.predict(X_test)
        acc   = accuracy_score(y_true, preds)
        pred_labels = ["Positive" if p == 1 else "Negative" for p in preds]
        pos_n = pred_labels.count("Positive")
        neg_n = pred_labels.count("Negative")

        with st.expander(f"📊 {model_name} — Accuracy: {acc*100:.1f}%"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc*100:.1f}%")
            c2.metric("Positive Predictions", pos_n)
            c3.metric("Negative Predictions", neg_n)

            # Classification report
            report = classification_report(
                y_true, preds,
                target_names=["Negative", "Positive"],
                output_dict=False
            )
            st.text("Classification Report:")
            st.code(report, language="text")

            # Confusion matrix
            cm = confusion_matrix(y_true, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            fig_cm.patch.set_facecolor("#1f2937")
            ax_cm.set_facecolor("#1f2937")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
            disp.plot(ax=ax_cm, colorbar=False, cmap="Greens")
            ax_cm.title.set_color("white")
            ax_cm.xaxis.label.set_color("white")
            ax_cm.yaxis.label.set_color("white")
            ax_cm.tick_params(colors="white")
            ax_cm.set_title(f"Confusion Matrix — {model_name}", color="white")
            st.pyplot(fig_cm)

            result_df = test_df[["text"]].copy()
            result_df["True Label"]  = ["Positive" if l == 1 else "Negative" for l in y_true]
            result_df["Prediction"]  = pred_labels
            result_df["Correct"]     = result_df["True Label"] == result_df["Prediction"]
            st.dataframe(result_df, use_container_width=True)

    st.subheader("Full Prediction Table (Logistic Regression)")
    preds_lr = model_lr.predict(X_test)
    # FIX: vectorized predict_proba — one call for all rows
    proba_lr = model_lr.predict_proba(X_test)
    final_df = test_df[["text"]].copy()
    final_df["True Label"]      = ["Positive" if l == 1 else "Negative" for l in y_true]
    final_df["Prediction"]      = ["Positive" if p == 1 else "Negative" for p in preds_lr]
    final_df["Confidence (%)"]  = [round(max(p) * 100, 1) for p in proba_lr]
    final_df["Correct"]         = final_df["True Label"] == final_df["Prediction"]
    st.dataframe(final_df, use_container_width=True)

# =========================================================
# SENTIMENT DEMO
# =========================================================

elif menu == "🎓 Sentiment Demo":
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("🎓 Online Learning Sentiment Analysis")

    st.write("""
    Enter a student's opinion about online learning and the model
    will predict whether the sentiment is Positive or Negative.
    """)

    col_model, _ = st.columns([1, 2])
    with col_model:
        chosen_model_name = st.selectbox(
            "Choose Model",
            ["Logistic Regression", "Naive Bayes", "SVM"]
        )

    chosen_model = {
        "Logistic Regression": model_lr,
        "Naive Bayes":         model_nb,
        "SVM":                 model_svm,
    }[chosen_model_name]

    user_input = st.text_area(
        "Enter Opinion",
        height=200,
        placeholder="Example: Saya merasa kuliah online sangat membantu karena lebih fleksibel."
    )

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned_text     = clean_text(user_input)
            transformed_text = vectorizer.transform([cleaned_text])
            prediction       = chosen_model.predict(transformed_text)[0]

            if hasattr(chosen_model, "predict_proba"):
                probabilities  = chosen_model.predict_proba(transformed_text)[0]
                negative_score = probabilities[0] * 100
                positive_score = probabilities[1] * 100
                confidence     = max(probabilities) * 100
            else:
                # LinearSVC: use decision_function as proxy
                decision       = chosen_model.decision_function(transformed_text)[0]
                positive_score = max(0.0, min(100.0, 50 + decision * 10))
                negative_score = 100 - positive_score
                confidence     = positive_score if prediction == 1 else negative_score

            st.write("")
            st.subheader("Prediction Result")

            if prediction == 1:
                st.markdown(f"""
                <div style="background-color:#14532d;padding:25px;border-radius:15px;
                            text-align:center;color:white;font-size:24px;font-weight:bold;">
                    😊 POSITIVE SENTIMENT<br><br>
                    Confidence: {confidence:.2f}%
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#7f1d1d;padding:25px;border-radius:15px;
                            text-align:center;color:white;font-size:24px;font-weight:bold;">
                    😞 NEGATIVE SENTIMENT<br><br>
                    Confidence: {confidence:.2f}%
                </div>""", unsafe_allow_html=True)

            st.write("")
            st.subheader("Confidence Scores")
            st.write(f"Negative Probability: {negative_score:.2f}%")
            st.progress(int(negative_score))
            st.write(f"Positive Probability: {positive_score:.2f}%")
            st.progress(int(positive_score))

            st.write("")
            st.subheader("Processed Text")
            st.code(cleaned_text, language="text")

            result_df = pd.DataFrame({
                "Original Text":   [user_input],
                "Processed Text":  [cleaned_text],
                "Prediction":      ["Positive" if prediction == 1 else "Negative"],
                "Confidence (%)":  [round(confidence, 2)]
            })
            st.subheader("Prediction Summary")
            st.dataframe(result_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
