import spacy
import json
from transformers import pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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


# Example usage
if __name__ == '__main__':
    # Create an instance of the pipeline
    pipeline = DynamicMedicalNLPPipeline()
    
    # Example transcript
    sample_transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    Patient: Yes, I always do.
    Physician: What did you feel immediately after the accident?
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Physician: Did you seek medical attention at that time?
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    Physician: How did things progress after that?
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    Physician: That makes sense. Are you still experiencing pain now?
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    [Physical Examination Conducted]
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    Patient: That's a relief!
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    Patient: Thank you, doctor. I appreciate it.
    Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """
    
    # Process the transcript
    result = pipeline.process_transcript(sample_transcript)
    
    # Print the result
    print(json.dumps(result, indent=2))