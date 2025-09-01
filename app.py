import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# Define CSS for Dark & Light Mode
# ------------------------
dark_theme = """
<style>
body, .stApp { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
h1,h2,h3,h4,h5 { color: #ffffff !important; }
.stButton>button { background-color: #2d2d2d; color: #ffffff; border-radius: 8px; border: 1px solid #3a3a3a; padding: 0.6em 1em; }
.stButton>button:hover { background-color: #3e3e3e; border: 1px solid #5a5a5a; }
.dataframe { background: #1e1e1e !important; color: #e0e0e0 !important; }
.stFileUploader { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333333; }
.stTabs [role="tab"][aria-selected="true"] { background: #2a2a2a; color: #ffffff; border-bottom: 3px solid #00adb5; }
</style>
"""

light_theme = """
<style>
body, .stApp { background-color: #ffffff; color: #1a1a1a; font-family: 'Segoe UI', sans-serif; }
h1,h2,h3,h4,h5 { color: #000000 !important; }
.stButton>button { background-color: #f0f0f0; color: #000000; border-radius: 8px; border: 1px solid #cccccc; padding: 0.6em 1em; }
.stButton>button:hover { background-color: #e0e0e0; border: 1px solid #999999; }
.dataframe { background: #ffffff !important; color: #000000 !important; }
.stFileUploader { background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #cccccc; }
.stTabs [role="tab"][aria-selected="true"] { background: #eeeeee; color: #000000; border-bottom: 3px solid #0077b6; }
</style>
"""

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Sidebar Theme Switch
theme_choice = st.sidebar.radio("🎨 Choose Theme", ["🌙 Dark", "☀️ Light"])
if theme_choice == "🌙 Dark":
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

st.title("📰 Fake News Detection App")
st.markdown("Detect whether a news article/headline is **Fake** or **Real** using Machine Learning.")

# ------------------------
# Load trained model
# ------------------------
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------
# Tabs for Input Modes
# ------------------------
tab1, tab2 = st.tabs(["🔍 Single News Check", "📂 Upload CSV Dataset"])

# (Rest of your app logic — input, prediction, history, clear button, CSV upload — remains same)


# ------------------------
# SINGLE NEWS CHECK TAB
# ------------------------
with tab1:
    st.subheader("🔍 Check a Single News Article")

    user_input = st.text_area("Paste News Article/Headline Here:", height=150)

    mode = st.radio("Choose Display Mode:", ["Simple Result", "Detailed Result (Probabilities)"], horizontal=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Check News", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before checking.")
        else:
            # Convert input
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            prediction_proba = model.predict_proba(input_vec)[0]

            fake_prob = round(prediction_proba[0] * 100, 2)
            real_prob = round(prediction_proba[1] * 100, 2)

            # --- Display Results ---
            if prediction == 0:
                st.error(f"❌ Fake News Detected!")
                confidence = fake_prob
                st.metric("Confidence Level", f"{fake_prob}% Fake")
            else:
                st.success(f"✅ Real News Detected!")
                confidence = real_prob
                st.metric("Confidence Level", f"{real_prob}% Real")

            if mode == "Detailed Result (Probabilities)":
                st.subheader("📊 Prediction Probabilities")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{fake_prob}%")
                with col2:
                    st.metric("Real Probability", f"{real_prob}%")

                # Chart
                labels = ["Fake", "Real"]
                values = [fake_prob, real_prob]
                fig, ax = plt.subplots()
                ax.bar(labels, values, color=["red", "green"], width=0.5)
                ax.set_ylim([0, 100])
                ax.set_ylabel("Probability (%)")
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)

            # Save history
            result = "Fake" if prediction == 0 else "Real"
            st.session_state.history.append(
                {"Text": user_input[:50] + ("..." if len(user_input) > 50 else ""),
                 "Prediction": result,
                 "Confidence": f"{confidence}%"}
            )
            if len(st.session_state.history) > 10:
                st.session_state.history.pop(0)

    # --- Show history ---
    if st.session_state.history:
        st.subheader("📝 Prediction History (Last 10)")
        st.table(st.session_state.history)

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.success("History cleared ✅")

# ------------------------
# UPLOAD CSV CHECK TAB
# ------------------------
with tab2:
    st.subheader("📂 Upload a CSV Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file (must have a `text` column)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("❌ CSV must contain a 'text' column!")
            else:
                st.success(f"✅ Loaded {len(df)} news articles from CSV.")

                if st.button("Run Batch Prediction", use_container_width=True):
                    input_vec = vectorizer.transform(df["text"].astype(str))
                    predictions = model.predict(input_vec)
                    probabilities = model.predict_proba(input_vec)

                    df["Prediction"] = ["Fake" if p == 0 else "Real" for p in predictions]
                    df["Fake Probability (%)"] = (probabilities[:, 0] * 100).round(2)
                    df["Real Probability (%)"] = (probabilities[:, 1] * 100).round(2)

                    # Show results
                    st.subheader("📊 Batch Prediction Results (first 20 rows)")
                    st.dataframe(df.head(20))

                    # Chart summary
                    fake_count = (df["Prediction"] == "Fake").sum()
                    real_count = (df["Prediction"] == "Real").sum()

                    st.subheader("📈 Fake vs Real Distribution")
                    fig, ax = plt.subplots()
                    ax.bar(["Fake", "Real"], [fake_count, real_count], color=["red", "green"], width=0.5)
                    ax.set_ylabel("Count")
                    ax.set_title("Fake vs Real News in Dataset")
                    st.pyplot(fig)

                    # Download option
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fake_news_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
