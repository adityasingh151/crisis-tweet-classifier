import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# === Page style ===
st.set_page_config(page_title="Crisis Tweet Classifier", page_icon="🚨", layout="centered")
# === Label map ===
LABEL_MAP = {
    0: "Other Useful Information",
    1: "Not Humanitarian",
    2: "Rescue/Volunteering/Donation Effort",
    3: "Injured or Dead People",
    4: "Not Related or Irrelevant",
    5: "Infrastructure and Utilities Damage",
    6: "Donation Needs or Offers or Volunteering Services",
    7: "Sympathy and Emotional Support",
    8: "Caution and Advice",
    9: "Affected People",
    10: "Affected Individuals",
    11: "Displaced People and Evacuations",
    12: "Missing, Trapped, or Found People",
    13: "Treatment",
    14: "Disease Signs or Symptoms",
    15: "Disease Transmission",
    16: "Prevention",
    17: "Deaths Reports",
    18: "Vehicle Damage"
}

# === Load model & tokenizer ===
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bertweet_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("bertweet_model")
    return tokenizer, model

tokenizer, model = load_model()



# === App title ===
st.title("🚨 Crisis Tweet Classifier (BERTweet)")
with st.expander("ℹ️ About this App"):
    st.markdown("""
**Crisis Tweet Classifier** is a lightweight AI tool designed to help **citizens, journalists, NGOs, and authorities** quickly understand the type of crisis-related information being shared on Twitter.

🌍 **Why this is useful:**  
During natural disasters, pandemics, or emergencies, millions of tweets are posted every hour. Sorting them by **humanitarian category** (e.g., donations, missing people, infrastructure damage) helps emergency teams:
- Detect urgent rescue needs.
- Prioritize resources.
- Identify offers of help and supplies.
- Monitor misinformation or irrelevant chatter.

✅ **Who can use this:**  
- **General public** can test their tweets to see how the AI understands them.  
- **Journalists** can quickly filter large tweet sets for stories.  
- **Disaster response teams** can combine this with other tools to triage social media data faster.  
- **Authorities & NGOs** can prototype how such AI helps in **situational awareness** — for example, during floods, earthquakes, or public health crises.

🔒 **Proven foundation:**  
This app uses a fine-tuned version of **BERTweet**, a trusted NLP model pre-trained on 850M English tweets.  
It’s built using **open-source transformers** and trained on trusted crisis datasets like **AIDR** and **CrisisMMD**, which have been cited in many academic studies on disaster informatics.

⚠️ **Important:**  
This AI is for **information support only**. It is not a replacement for official communication, verified rescue coordination, or medical guidance. Always verify critical crisis information with **reliable local authorities** and trusted news channels.

⚠️ **Only for Disaster-Related Tweets:**  
    This model is specifically trained on **disaster and crisis tweets**.  
    If you enter unrelated or everyday tweets, it will still try to predict them as a **crisis-related category**, which will give misleading results.  
    Please **only use real disaster or emergency context tweets** for meaningful predictions.
""")

st.markdown(
    "🔍 **Identify the type of crisis information from tweets instantly!**\n\n"
    "This tool uses a fine-tuned BERTweet model to detect humanitarian categories in tweets."
)

st.markdown("---")

# === SINGLE TWEET ===
st.header("📝 Single Tweet Prediction")

tweet_input = st.text_area(
    "✏️ Enter your tweet below:", 
    placeholder="Example: Floods have damaged the bridge near my village."
)

if st.button("🚀 Predict"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting!")
    else:
        with st.spinner("Thinking..."):
            inputs = tokenizer(tweet_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = probs.argmax().item()
            predicted_label_name = LABEL_MAP.get(predicted_label, "Unknown")

        st.success(f"✅ **Predicted Label:** {predicted_label_name}")
        st.info(f"🔢 **Label ID:** {predicted_label} | 💯 **Confidence:** {probs.max().item():.2f}")

st.markdown("---")

# === Batch Prediction ===
st.header("📂 Batch Prediction (CSV)")

with st.expander("📋 Click for CSV Upload Instructions"):
    st.write("""
    ✅ **Required format:**
    - Your file must have **one column named `Content`**.
    - Each row in `Content` should contain a single tweet.
    - Remove unrelated columns (IDs, timestamps, usernames).
    - Save as UTF-8 `.csv`.

    **Example CSV:**

    | Content |
    |------------------------------|
    | Floods have damaged bridges. |
    | Rescue teams needed urgently. |
    | Many people displaced. |
    """)

uploaded_file = st.file_uploader("📤 Upload your CSV file:", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "Content" not in df.columns:
            st.error("❌ The CSV must contain a `Content` column!")
        else:
            st.write("👀 **Preview of your file:**")
            st.dataframe(df.head())

            with st.spinner("Classifying tweets... please wait ⏳"):
                predictions = []
                confidences = []
                for text in df["Content"]:
                    if pd.isna(text) or str(text).strip() == "":
                        predictions.append("Empty")
                        confidences.append(0.0)
                        continue
                    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    label_id = probs.argmax().item()
                    label_name = LABEL_MAP.get(label_id, "Unknown")
                    predictions.append(label_name)
                    confidences.append(probs.max().item())

                df["Predicted_Label"] = predictions
                df["Confidence"] = confidences

            st.balloons()
            st.success("✅ Done! Check your results below 👇")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="predicted_tweets.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(
            f"❌ Something went wrong: {str(e)}\n\n"
            "Please ensure your CSV is properly formatted and encoded as UTF-8."
        )

# === Footer ===
st.markdown("---")
st.write("Made with ❤️ by **Aditya**")
