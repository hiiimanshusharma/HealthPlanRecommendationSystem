from transformers import pipeline
import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# Model paths
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH ='model/tfidfvectorizer.pkl'
DATA_PATH ='data/drugsComTrain_raw.csv'

# Load vectorizer and model
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

# Load stopwords and lemmatizer
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Function to clean text
def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. Lowercase letters
    words = letters_only.lower().split()
    # 5. Remove stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. Lemmatization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. Join words with spaces
    return ' '.join(lemmitize_words)

# Function to extract top drugs
def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

# Load a pre-trained language model (e.g., GPT-2)
generator = pipeline("text-generation", model="gpt2")

def generate_health_plan(medical_condition, medicine_names):
    prompt = f"Patient has {medical_condition} and is prescribed {', '.join(medicine_names)}. The health plan is as follows:"

    # Generate health plan text
    health_plan = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)[0]['generated_text']

    return health_plan

# Streamlit app
def main():
    st.title("Health Plan Recommendation")

    raw_text = st.text_area("Enter a symptons/test reports:", "")

    if st.button("Predict"):
        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]

            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)

            st.write("Predicted Condition:", predicted_cond)
            st.write("Top Drugs for this Condition:", top_drugs)

            # Generate health plan
            health_plan = generate_health_plan(predicted_cond, top_drugs)
            st.subheader("Health Plan:")
            st.write(health_plan)  # Display the generated health plan

        else:
            st.warning("Please enter a review text.")

if __name__ == "__main__":
    main()
