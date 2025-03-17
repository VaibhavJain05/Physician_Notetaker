import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
sentiment_labels = ["Anxious", "Neutral", "Reassured"]
intent_labels = ["Seeking Reassurance", "Reporting Symptoms", "Expressing Concern"]
num_sentiments = len(sentiment_labels)
num_intents = len(intent_labels)
num_labels = num_sentiments + num_intents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = AutoModelForSequenceClassification.from_pretrained("./saved_model").to(device)
model.eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def analyze_patient_sentiment_and_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    sentiment_pred = sentiment_labels[np.argmax(probs[:num_sentiments])]
    intent_pred = intent_labels[np.argmax(probs[num_sentiments:])]

    return {"Sentiment": sentiment_pred, "Intent": intent_pred}


st.title("Patient Sentiment & Intent Analysis")
st.write("Enter patient dialogue below:")

user_input = st.text_area("Patient Transcript:", "")

def batch_inference(transcript):
    results = {}
    for i, line in enumerate(transcript.split("\n")):
        line = line.strip()
        if line:
            results[f"Line {i+1}"] = analyze_patient_sentiment_and_intent(line)
    return results

if st.button("Analyze Transcript"):
    transcript_results = batch_inference(user_input)
    st.subheader("Transcript Analysis")
    st.write(transcript_results)



