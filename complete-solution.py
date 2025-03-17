import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from SOAP_generator import SOAPNoteGenerator

class DynamicMedicalNLPPipeline:
    def __init__(self, spacy_model="en_core_web_lg", custom_entities=None):
        """
        Initialize the pipeline with configurable parameters
        
        Args:
            spacy_model: The SpaCy model to use
            custom_entities: Custom entity categories and keywords
        """
        # Load SpaCy model
        self.nlp = spacy.load(spacy_model)
        
        # Initialize default entity categories if not provided
        self.custom_entities = custom_entities or {
            "SYMPTOM": ["pain", "discomfort", "aches", "stiffness", "trouble sleeping"],
            "BODY_PART": ["neck", "back", "head", "spine", "muscles"],
            "TREATMENT": ["physiotherapy", "painkillers", "X-rays"],
            "DIAGNOSIS": ["whiplash injury"],
            "TIME": [],
            "LOCATION": []
        }
                
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        
    def update_entity_categories(self, new_categories):
        """Dynamically update entity categories"""
        for category, keywords in new_categories.items():
            if category in self.custom_entities:
                self.custom_entities[category].extend(keywords)
                # Remove duplicates
                self.custom_entities[category] = list(set(self.custom_entities[category]))
            else:
                self.custom_entities[category] = keywords
    
    
    def extract_entities(self, text):
        """Extract medical entities from text using SpaCy and custom rules"""
        doc = self.nlp(text)
        
        # Initialize entity categories
        entities = {category: [] for category in self.custom_entities.keys()}
        
        # Extract named entities from SpaCy
        for ent in doc.ents:
            if ent.label_ == "DATE" or ent.label_ == "TIME":
                entities["TIME"].append(ent.text)
            elif ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ == "FAC":
                entities["LOCATION"].append(ent.text)
        
        # Custom rule-based entity extraction
        for category, keywords in self.custom_entities.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Find the complete phrase containing the keyword
                    pattern = r"[^.!?]*\b" + re.escape(keyword.lower()) + r"\b[^.!?]*[.!?]?"
                    matches = re.findall(pattern, text.lower())
                    
                    for match in matches:
                        # Clean and add the phrase
                        phrase = match.strip()
                        if phrase and keyword.lower() in phrase:
                            if keyword not in entities[category]:
                                entities[category].append(keyword)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
        
        return entities
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract important keywords using TF-IDF"""
        # If we only have one document, we need to create a corpus
        corpus = [text]
        
        # Fit and transform the text
        try:
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get importance scores
            scores = tfidf_matrix.toarray()[0]
            
            # Sort keywords by importance
            sorted_indices = np.argsort(scores)[::-1]
            
            # Get top keywords
            top_keywords = [feature_names[i] for i in sorted_indices[:num_keywords] if scores[i] > 0]
            return top_keywords
        except:
            # If TF-IDF fails (e.g., not enough documents), use simple word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:num_keywords]]
    
    def generate_medical_summary(self, transcript):
        """Generate a structured medical summary from the transcript"""
                
        # Extract entities
        entities = self.extract_entities(transcript)
        
        # Extract keywords
        keywords = self.extract_keywords(transcript)
        
        # Extract patient information
        patient_name = "Unknown"
        name_match = re.search(r"(?:Mr\.|Mrs\.|Ms\.|Dr\.) ([A-Z][a-z]+)", transcript)
        if name_match:
            patient_name = name_match.group(0)
        
        # Extract incident details
        incident_pattern = r"(car accident|fall|injury|incident).*?[.!?]"
        incident_match = re.search(incident_pattern, transcript, re.IGNORECASE)
        incident_description = incident_match.group(0) if incident_match else "Unknown incident"
        
        # Extract symptoms
        symptoms_current = []
        symptoms_pattern = r"(experiencing|feel|having|suffering from) (.*?)[.!?]"
        symptoms_matches = re.findall(symptoms_pattern, transcript, re.IGNORECASE)
        if symptoms_matches:
            symptoms_current = [match[1].strip() for match in symptoms_matches]
        
        # Extract prognosis
        prognosis_pattern = r"(expect|prognosis|outlook|recovery|future) (.*?)[.!?]"
        prognosis_match = re.search(prognosis_pattern, transcript, re.IGNORECASE)
        prognosis = prognosis_match.group(0) if prognosis_match else "Prognosis unknown"
        
        # Create structured summary
        summary = {
            "patient_info": {
                "name": patient_name
            },
            "incident": incident_description,
            "symptoms": {
                "current": symptoms_current if symptoms_current else "Not explicitly stated"
            },
            "diagnosis": entities.get("DIAGNOSIS", []),
            "treatments": entities.get("TREATMENT", []),
            "affected_body_parts": entities.get("BODY_PART", []),
            "prognosis": prognosis,
            "keywords": keywords
        }
        
        return summary

    
    def process_transcript(self, transcript):
        """Process a medical transcript and extract all relevant information"""
        # Generate medical summary
        medical_summary = self.generate_medical_summary(transcript)
                
        # Combine all results
        result = {
            "medical_summary": medical_summary,
        }
        
        return result

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

if st.button("Summarize Transcript"):
    pipeline = DynamicMedicalNLPPipeline()
    transcript_summary = pipeline.process_transcript(user_input)
    st.subheader("Transcript Summary")
    st.write(transcript_summary)

if st.button("SOAP"):
    generator = SOAPNoteGenerator()
    soap_note_json = generator.generate_soap_note_json(user_input)
    st.subheader("Transcript SOAP")
    st.write(soap_note_json)

    