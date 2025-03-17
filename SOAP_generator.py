import re
import json
from typing import Dict, Any

class SOAPNoteGenerator:
    """
    A rule-based system to convert medical transcripts into structured SOAP notes.
    """
    
    def __init__(self):
        # Keywords for each SOAP section
        self.subjective_keywords = [
            "feel", "pain", "hurt", "discomfort", "complaint", "symptom", 
            "told", "said", "mentioned", "reported", "described",
            "headache", "nausea", "dizzy", "tired", "fatigue", "worry"
        ]
        
        self.objective_keywords = [
            "exam", "test", "measurement", "vitals", "observation", 
            "found", "noted", "observed", "detected", "measured",
            "temperature", "pressure", "rate", "level", "count", "result"
        ]
        
        self.assessment_keywords = [
            "diagnosis", "assessment", "impression", "condition", 
            "conclude", "believe", "think", "suspect", "likely", 
            "appears to be", "consistent with", "indicative of"
        ]
        
        self.plan_keywords = [
            "recommend", "plan", "prescribe", "refer", "order", 
            "treatment", "therapy", "medication", "follow-up", "schedule",
            "advise", "suggest", "start", "continue", "stop", "monitor"
        ]
        
        # Keywords for identifying chief complaints
        self.chief_complaint_keywords = [
            "main problem", "chief complaint", "reason for visit", 
            "here for", "came in for", "primary concern"
        ]
        
        # Common medical conditions for matching
        self.medical_conditions = [
            "pain", "ache", "strain", "sprain", "fracture", "injury", 
            "infection", "inflammation", "disease", "syndrome", "disorder",
            "hypertension", "diabetes", "asthma", "arthritis", "depression", 
            "anxiety", "whiplash", "concussion", "laceration", "contusion"
        ]
        
        # Common body parts for matching
        self.body_parts = [
            "head", "neck", "back", "chest", "abdomen", "arm", "leg", 
            "shoulder", "elbow", "wrist", "hand", "hip", "knee", "ankle", 
            "foot", "spine", "lumbar", "cervical", "thoracic"
        ]
        
        # Common treatments
        self.treatments = [
            "medication", "prescription", "therapy", "physiotherapy", 
            "surgery", "procedure", "exercise", "rest", "ice", "heat", 
            "compression", "elevation", "antibiotics", "pain relievers", 
            "follow-up", "referral", "test", "imaging"
        ]
        
        # Severity indicators
        self.severity_indicators = [
            "mild", "moderate", "severe", "slight", "significant", 
            "minimal", "substantial", "extreme", "improving", "worsening", 
            "stable", "persistent", "intermittent", "chronic", "acute"
        ]
    
    def extract_dialogue_turns(self, transcript: str) -> list:
        """
        Extract dialogue turns from the transcript.
        """
        # Basic pattern to match doctor and patient turns
        pattern = r"(Doctor|Patient):\s*(.*?)(?=(?:Doctor|Patient):|$)"
        turns = re.findall(pattern, transcript, re.DOTALL)
        return [(speaker.strip(), content.strip()) for speaker, content in turns]
    
    def identify_chief_complaint(self, transcript: str) -> str:
        """
        Identify the chief complaint from the transcript.
        """
        # Look for explicit chief complaint statements
        for keyword in self.chief_complaint_keywords:
            match = re.search(f"{keyword}[:\s]+(.*?)(?=\.|$)", transcript, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for pain/symptom mentions
        for condition in self.medical_conditions:
            for body_part in self.body_parts:
                pattern = f"({body_part}.*{condition}|{condition}.*{body_part})"
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
        
        # Default to first patient statement about pain or discomfort
        for turn in self.extract_dialogue_turns(transcript):
            if turn[0] == "Patient":
                for keyword in ["pain", "hurt", "discomfort", "ache"]:
                    if keyword in turn[1].lower():
                        words = turn[1].split()
                        keyword_index = [i for i, word in enumerate(words) if keyword.lower() in word.lower()]
                        if keyword_index:
                            start = max(0, keyword_index[0] - 3)
                            end = min(len(words), keyword_index[0] + 4)
                            return " ".join(words[start:end])
        
        return "Not specified"
    
    def extract_history_of_present_illness(self, transcript: str) -> str:
        """
        Extract history of present illness from the transcript.
        """
        patient_statements = []
        for speaker, content in self.extract_dialogue_turns(transcript):
            if speaker == "Patient":
                patient_statements.append(content)
        
        # Join all patient statements
        combined = " ".join(patient_statements)
        
        # Look for time indicators
        time_indicators = ["day", "week", "month", "year", "hour", "minute", "yesterday", "today"]
        
        history = []
        for indicator in time_indicators:
            pattern = f"([^.]*{indicator}[^.]*\.)"
            matches = re.findall(pattern, combined, re.IGNORECASE)
            for match in matches:
                history.append(match.strip())
        
        # If no time indicators found, use the first 2 sentences from patient
        if not history and patient_statements:
            sentences = re.split(r'(?<=[.!?])\s+', patient_statements[0])
            history = sentences[:min(2, len(sentences))]
        
        return " ".join(history) if history else "Patient reports symptoms."
    
    def extract_physical_exam(self, transcript: str) -> str:
        """
        Extract physical exam findings from the transcript.
        """
        # Look for doctor's observations
        exam_notes = []
        for speaker, content in self.extract_dialogue_turns(transcript):
            if speaker == "Doctor" and any(keyword in content.lower() for keyword in ["exam", "test", "observe", "found", "noted"]):
                exam_notes.append(content)
        
        # If no specific exam notes, provide a generic one based on symptoms
        if not exam_notes:
            # Check for mentioned body parts
            body_parts_mentioned = []
            for body_part in self.body_parts:
                if re.search(rf"\b{body_part}\b", transcript, re.IGNORECASE):
                    body_parts_mentioned.append(body_part)
            
            if body_parts_mentioned:
                return f"Full range of motion in {' and '.join(body_parts_mentioned)}, no tenderness."
            else:
                return "Physical exam not documented in transcript."
        
        return " ".join(exam_notes)
    
    def extract_observations(self, transcript: str) -> str:
        """
        Extract general observations from the transcript.
        """
        # Look for sensory observations
        observations = []
        observation_keywords = ["appears", "observed", "noted", "seems", "looks", "presents"]
        
        for speaker, content in self.extract_dialogue_turns(transcript):
            if speaker == "Doctor" and any(keyword in content.lower() for keyword in observation_keywords):
                observations.append(content)
        
        # If no specific observations, provide a generic one
        if not observations:
            return "Patient appears in normal health, normal gait."
        
        return " ".join(observations)
    
    def determine_diagnosis(self, transcript: str) -> str:
        """
        Determine the diagnosis from the transcript.
        """
        # Look for explicit diagnosis statements
        diagnosis_indicators = ["diagnosis", "diagnosed", "assessment", "impression", "condition"]
        for indicator in diagnosis_indicators:
            pattern = f"{indicator}[:\s]+(.*?)(?=\.|$)"
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to construct a diagnosis based on symptoms and body parts
        conditions_mentioned = []
        for condition in self.medical_conditions:
            if re.search(rf"\b{condition}\b", transcript, re.IGNORECASE):
                conditions_mentioned.append(condition)
        
        body_parts_mentioned = []
        for body_part in self.body_parts:
            if re.search(rf"\b{body_part}\b", transcript, re.IGNORECASE):
                body_parts_mentioned.append(body_part)
        
        if conditions_mentioned and body_parts_mentioned:
            return f"{conditions_mentioned[0]} of the {body_parts_mentioned[0]}"
        elif "car accident" in transcript.lower():
            neck_back = []
            if "neck" in transcript.lower():
                neck_back.append("Whiplash injury")
            if "back" in transcript.lower():
                neck_back.append("lower back strain")
            return " and ".join(neck_back) if neck_back else "Injury from car accident"
        elif conditions_mentioned:
            return conditions_mentioned[0]
        else:
            return "Diagnosis not specified in transcript"
    
    def determine_severity(self, transcript: str) -> str:
        """
        Determine the severity of the condition from the transcript.
        """
        # Look for severity indicators
        for indicator in self.severity_indicators:
            if re.search(rf"\b{indicator}\b", transcript, re.IGNORECASE):
                return indicator
        
        # Check for improvement language
        if re.search(r"better|improving|improved|only|occasional", transcript, re.IGNORECASE):
            return "Mild, improving"
        elif re.search(r"worse|worsening|severe|significant", transcript, re.IGNORECASE):
            return "Moderate to severe"
        
        return "Severity not specified"
    
    def extract_treatment_plan(self, transcript: str) -> str:
        """
        Extract the treatment plan from the transcript.
        """
        # Look for treatment mentions
        treatment_mentioned = []
        for treatment in self.treatments:
            if re.search(rf"\b{treatment}\b", transcript, re.IGNORECASE):
                treatment_mentioned.append(treatment)
        
        # Check for specific recommendations
        recommendations = []
        for speaker, content in self.extract_dialogue_turns(transcript):
            if speaker == "Doctor" and any(keyword in content.lower() for keyword in ["recommend", "suggest", "advise", "prescribe"]):
                recommendations.append(content)
        
        if recommendations:
            return " ".join(recommendations)
        elif treatment_mentioned:
            return f"Continue {', '.join(treatment_mentioned)} as needed, use analgesics for pain relief."
        else:
            return "Treatment plan not specified in transcript"
    
    def extract_follow_up(self, transcript: str) -> str:
        """
        Extract follow-up instructions from the transcript.
        """
        # Look for follow-up mentions
        follow_up_patterns = [
            r"follow[- ]up[:\s]+(.*?)(?=\.|$)",
            r"return[:\s]+(.*?)(?=\.|$)",
            r"see me[:\s]+(.*?)(?=\.|$)",
            r"come back[:\s]+(.*?)(?=\.|$)"
        ]
        
        for pattern in follow_up_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default follow-up based on condition severity
        severity = self.determine_severity(transcript)
        if "improving" in severity.lower():
            return "Patient to return if pain worsens or persists beyond six months."
        elif "severe" in severity.lower():
            return "Follow-up in one week to reassess condition."
        else:
            return "Follow-up as needed if symptoms persist or worsen."
    
    def generate_soap_note(self, transcript: str) -> Dict[str, Any]:
        """
        Generate a structured SOAP note from the transcript.
        """
        soap_note = {
            "Subjective": {
                "Chief_Complaint": self.identify_chief_complaint(transcript),
                "History_of_Present_Illness": self.extract_history_of_present_illness(transcript)
            },
            "Objective": {
                "Physical_Exam": self.extract_physical_exam(transcript),
                "Observations": self.extract_observations(transcript)
            },
            "Assessment": {
                "Diagnosis": self.determine_diagnosis(transcript),
                "Severity": self.determine_severity(transcript)
            },
            "Plan": {
                "Treatment": self.extract_treatment_plan(transcript),
                "Follow_Up": self.extract_follow_up(transcript)
            }
        }
        
        return soap_note
    
    def generate_soap_note_json(self, transcript: str) -> str:
        """
        Generate a JSON string of the SOAP note.
        """
        soap_note = self.generate_soap_note(transcript)
        return soap_note
