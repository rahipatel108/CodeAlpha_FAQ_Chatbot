import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

st.write("Ask me a question!")

question = st.text_input("Enter your question:")

faq = pd.read_csv("faq.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = text.split()
    return " ".join(tokens)

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

     questions = [preprocess(q) for q in faq["Question"]]
     user_question = preprocess(question)

     vectorizer = TfidfVectorizer()

     vectors = vectorizer.fit_transform(questions + [user_question])

     similarity = cosine_similarity(vectors[-1], vectors[:-1])

     best_match = similarity.argmax()

     if similarity[0][best_match] > 0.3:
        st.success(faq["Answer"][best_match])
     else:
        st.error("Sorry, I don't know the answer.")
        