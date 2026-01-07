import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from datetime import datetime
from reportlab.lib.utils import ImageReader
import smtplib
from email.message import EmailMessage
import speech_recognition as sr
import re
import pyttsx3
import time
import os
import uuid
import shutil
from textwrap import wrap
import pdfplumber
import pytesseract
from PIL import Image
import random

if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'proba' not in st.session_state:
    st.session_state.proba = None
if 'shap_df' not in st.session_state:
    st.session_state.shap_df = None
if 'shap_fig' not in st.session_state:
    st.session_state.shap_fig = None
if 'lab_values' not in st.session_state:
    st.session_state.lab_values = None


def generate_patient_id(patient_name):
    if not patient_name:
        return None
        
    csv_path = "patient_history.csv"
    
    if not os.path.exists(csv_path):
        return f"PCOS-{datetime.now().year}-001"
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return f"PCOS-{datetime.now().year}-001"
    
    last_id = df['Patient_ID'].iloc[-1].split("-")[-1]
    new_number = int(last_id) + 1
    return f"PCOS-{datetime.now().year}-{new_number:03d}"

LOW_GI_FOODS = {
    "proteins": ["Salmon", "Lentils", "Tofu", "Chicken", "Greek yogurt"],
    "carbs": ["Quinoa", "Sweet potato", "Berries", "Oats", "Whole wheat pasta"],
    "veggies": ["Broccoli", "Spinach", "Bell peppers", "Zucchini"],
    "fats": ["Avocado", "Almonds", "Olive oil", "Chia seeds"]
}

MEAL_TEMPLATES = {
    "breakfast": [
        "Oats with {nuts} and cinnamon",
        "Greek yogurt with {berries}",
        "Scrambled tofu with {veggies}"
    ],
    "lunch": [
        "Quinoa salad with {protein} and {veggies}",
        "Whole wheat wrap with {protein} and avocado"
    ],
    "dinner": [
        "Grilled {protein} with roasted {veggies}",
        "Stir-fried tofu with {veggies} over brown rice"
    ]
}
WORKOUT_PLANS = {
    "beginner": {
        "cardio": ["Brisk walking 30min", "Cycling 20min", "Swimming 15min"],
        "strength": ["Bodyweight squats (3x10)", "Wall push-ups (3x8)", "Seated rows (2x10)"]
    },
    "intermediate": {
        "cardio": ["Jogging 25min", "Jump rope 15min", "Dance workout 20min"],
        "strength": ["Lunges (3x12)", "Push-ups (3x10)", "Dumbbell rows (3x12)"]
    }
}
EXERCISE_TIPS = {
    "high_glucose": "Do 10min walks after meals to lower blood sugar spikes",
    "high_bmi": "Combine cardio and strength training for optimal fat loss",
    "high_stress": "Yoga reduces cortisol levels by 30%"
}

MENTAL_ACTIVITIES = {
    "high_stress": [
        "4-7-8 Breathing (inhale 4s, hold 7s, exhale 8s)",
        "Progressive Muscle Relaxation",
        "Guided Imagery Meditation"
    ],
    "moderate_stress": [
        "5-Minute Journaling",
        "Nature Walk (no phone!)",
        "Box Breathing (4x4x4)"
    ]
}
SLEEP_TIPS = {
    "poor_sleep": [
        "Keep bedroom temperature between 60-67°F",
        "Avoid screens 1 hour before bed",
        "Try white noise or earplugs"
    ]
}
MEDICAL_CHECKS = {
    "high_hba1c": [
        "Schedule an A1C test every 3 months",
        "Ask your doctor about a Continuous Glucose Monitor (CGM)"
    ],
    "high_bmi": [
        "Request a thyroid function test (TSH, T3, T4)",
        "Discuss metabolic syndrome screening"
    ],
    "family_history": [
        "Annual eye exam for diabetic retinopathy",
        "Foot neuropathy check every 6 months"
    ]
}
DOCTOR_QUESTIONS = {
    "prediabetes": [
        "Should I start metformin for prevention?",
        "What's my ideal fasting glucose target?"
    ],
    "high_cholesterol": [
        "Do I need a statin?",
        "How often should I check my LDL?"
    ]
}

def extract_values_from_report(file):
    """Process uploaded reports using OCR/text extraction"""
    text = ""
    
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    else:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
    
    return text



def parse_medical_values(text):
   
    patterns = {
        'HbA1c': r"(HbA1c.*?|Glycosylated Hemoglobin.*?|HBA1C.*?)(\d+\.\d+)",
        'Fasting_Glucose': r"(Glucose, Fasting|Fasting Glucose|FBS|Blood Sugar).*?(\d+\.\d+)",
        'Post_Meal_Glucose': r"(Post[-\s]?[Pp]randial Glucose|Glucose\s*Post[-\s]?[Pp]randial|PPG|2[- ]?hr Glucose|2[- ]?hPP|Post Meal Glucose)[\s:]*(\d+\.?\d*)",
        'Cholesterol': r"(Total Cholesterol|CHOL).*?(\d+\.\d+)",
        'C_Peptide': r"(C[\s\-]?Peptide.*?)(\d+\.\d+)",
        'Ketones': r"(Ketones.*?)(\d+\.\d+)"
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                extracted[key] = float(match.group(2))
            except ValueError:
                continue
    return extracted

def parse_patient_demographics(text):
    demographics = {}
    name_patterns = [
        r"Patient Name:\s*([A-Za-z, ]+)",
        r"Name:\s*([A-Za-z, ]+)",
        r"Patient:\s*([A-Za-z, ]+)",
        r"PATIENT NAME\s*([A-Za-z, ]+)"
    ]
    age_patterns = [
        r"Age:\s*(\d+)",
        r"Age\/Sex:\s*(\d+)",
        r"Patient Age:\s*(\d+)",
        r"AGE\s*(\d+)"
    ]
    sex_patterns = [
        r"Sex:\s*([MF])",
        r"Gender:\s*([MF])",
        r"Age\/Sex:\s*\d+\/([MF])",
        r"SEX\s*([MF])"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            demographics['Name'] = match.group(1).strip()
            break
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                demographics['Age'] = int(match.group(1))
            except ValueError:
                continue
            break
    for pattern in sex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sex = match.group(1).upper()
            demographics['Sex'] = "Male" if sex == "M" else "Female"
            break
    return demographics


model = joblib.load("best_diabetes_model.pkl")
    


model_features = [
    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 
    'BP_Systolic', 'BP_Diastolic', 'Cholesterol', 'Stress', 
    'Hereditary', 'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose', 
    'C_Peptide', 'Ketones'
]

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🩸 Diabetes Risk Predictor")
st.warning("⚠️ Important: This predictor is only accurate if you are not currently taking any diabetes medications.")
st.markdown("""
    Enter your health details to check diabetes risk. 
    All predictions are AI-based and should be followed up with a doctor's opinion.
""")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts diabetes risk (No Diabetes, Pre-Diabetes, Type 1-Diabetes or Type 2-Diabetes) 
    using machine learning based on your health parameters.
    """)
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("- Fill in all health metrics")
    st.markdown("- Click 'Predict Diabetes' for results")
    st.markdown("- For Gender: M = Male, F = Female")
    st.markdown("- Stress: None, Low, Medium, High")

st.subheader("🩸 Alternative: Upload Blood Report")
analysis_choice = st.radio(
    "Choose input method:",
    ["✍️ Manual Entry", "📄 Upload Lab Report"],
    horizontal=True
)

if analysis_choice == "📄 Upload Lab Report":
    uploaded_files = st.file_uploader(
        "Upload blood test report (PDF/JPEG/PNG)", 
        type=["pdf", "jpg", "jpeg", "png"],
        help="Ensure values for HbA1c, Glucose, etc. are visible",
        accept_multiple_files=True
    )
    if uploaded_files:
        all_lab_values = {}
        for uploaded_file in uploaded_files:
            with st.spinner("🔍 Analyzing your report..."):
                try:
                    raw_text = extract_values_from_report(uploaded_file)
                    
                    lab_values = parse_medical_values(raw_text)
                    demographics = parse_patient_demographics(raw_text)
                    
                    st.session_state.lab_values = lab_values
                    
                    if 'Name' in demographics:
                        st.session_state.report_name = demographics['Name']
                    if 'Age' in demographics:
                        st.session_state.Age = float(demographics['Age'])
                    if 'Sex' in demographics:
                        st.session_state.Gender = 1 if demographics['Sex'] == "Male" else 0
                    
                    st.success("✅ Found these values in your report:")
                    st.json(lab_values)
                    
                    if demographics:
                        st.info("👤 Extracted Patient Details:")
                        st.json(demographics)
                    
                    st.markdown("**📊 Extracted Values vs Normal Ranges:**")
                    ranges = {
                        'HbA1c': (4.0, 5.6, 6.4),
                        'Fasting_Glucose': (70, 99, 125),
                        'Post_Meal_Glucose': (70, 140, 200),
                        'Cholesterol': (0, 200, 240),
                        'C_Peptide': (1.1, 4.4, 8.0),
                        'Ketones': (0.0, 0.6, 3.0)
                    }
                    
                    for param, value in lab_values.items():
                        if param in ranges:
                            low, normal, high = ranges[param]
                            status = "🔴 High" if value >= high else "🟡 Warning" if value >= normal else "🟢 Normal"
                            st.metric(
                                label=f"{param}: {value}",
                                value=status,
                                help=f"Normal range: {low}-{normal}"
                            )
                    
                    missing = set(model_features) - set(lab_values.keys()) - {'Gender', 'Stress', 'Hereditary', 'Age'}
                    if missing:
                        st.warning(f"⚠️ Could not find: {', '.join(missing)}. Please enter manually below.")
                
                except Exception as e:
                    st.error(f"❌ Report processing failed: {str(e)}")

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech error: {str(e)}")

def listen():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 100 

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("Listening... (speak now)")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            try:
                text = recognizer.recognize_google(audio)
                st.success(f"Recognized: {text}")
                return text.lower()
            except sr.UnknownValueError:
                st.warning("Could not understand audio. Please try again.")
                return ""
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {str(e)}")
                return ""
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. Please try again.")
            return ""
        except Exception as e:
            st.error(f"Microphone error: {str(e)}")
            return ""

if st.button("🎤 Start Voice Input"):
    speak("Welcome to Diabetes Voice Assistant. I will ask you for each health parameter.")
    
    voice_data = {}
    for feature in model_features:
        clean_name = feature.replace("_", " ").title()
        
        if feature == "Gender":
            speak(f"Please say your gender - Male or Female")
            gender = listen()
            if "male" in gender:
                voice_data[feature] = 1
            elif "female" in gender:
                voice_data[feature] = 0
            else:
                st.warning("Could not recognize gender. Please enter manually.")
                continue
                
        elif feature == "Stress":
            speak("Please say your stress level - None, Low, Medium, or High")
            stress = listen()
            if "none" in stress:
                voice_data[feature] = 0
            elif "low" in stress:
                voice_data[feature] = 1
            elif "medium" in stress:
                voice_data[feature] = 2
            elif "high" in stress:
                voice_data[feature] = 3
            else:
                st.warning("Could not recognize stress level. Please enter manually.")
                continue
                
        elif feature == "Hereditary":
            speak("Does diabetes run in your family? Say Yes or No")
            hereditary = listen()
            if "yes" in hereditary:
                voice_data[feature] = 1
            elif "no" in hereditary:
                voice_data[feature] = 0
            else:
                st.warning("Could not recognize answer. Please enter manually.")
                continue
                
        else:
            speak(f"Please say your {clean_name}")
            value = listen()
            try:
                num = float(re.search(r'\d+\.?\d*', value).group())
                voice_data[feature] = num
            except:
                st.warning(f"Could not recognize {clean_name}. Please enter manually.")
                continue
                
    for feature, value in voice_data.items():
        st.session_state[feature] = value
    st.success("Voice input completed!")

if st.button("🔁 Reset Voice Inputs"):
    speak("Would you like to reset a specific field or all?")
    choice = listen().lower()

    if choice:
        if "all" in choice:
            for f in model_features:
                st.session_state.pop(f, None)
            st.success("✅ All inputs reset.")
            speak("All inputs reset.")
        else:
            speak("Which field to reset?")
            target = listen().lower()
            if target:
                found = False
                for f in model_features:
                    clean = f.replace("_", " ").replace("(y/n)", "").replace("(r/i)", "").strip().lower()
                    if target in clean:
                        st.session_state.pop(f, None)
                        st.success(f"{clean.title()} reset.")
                        speak(f"{clean} reset.")
                        found = True
                        break
                if not found:
                    st.warning("❌ Field not found.")
                    speak("Field not found.")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    st.subheader("Basic Information")
    user_input['Age'] = st.number_input("Age (years)", min_value=1.0, max_value=120.0, 
                                      value=float(st.session_state.get('Age', 30)))
    user_input['Gender'] = st.radio("Gender", ["Female", "Male"], 
                                   index=st.session_state.get('Gender', 0))
    user_input['Height_cm'] = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=0.1,
                                            value=float(st.session_state.get('Height_cm', 170.0)))
    user_input['Weight_kg'] = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, step=0.1,
                                            value=float(st.session_state.get('Weight_kg', 70.0)))
    # Auto-calculate BMI
    if user_input['Height_cm'] > 0 and user_input['Weight_kg'] > 0:
        user_input['BMI'] = round(user_input['Weight_kg'] / ((user_input['Height_cm']/100) ** 2), 1)
        st.metric("Calculated BMI", user_input['BMI'])
    else:
        user_input['BMI'] = 22.0

with col2:
    st.subheader("Health Metrics")
    user_input['BP_Systolic'] = st.number_input("BP Systolic", min_value=80.0, max_value=200.0,
                                              value=float(st.session_state.get('BP_Systolic', 120.0)))
    user_input['BP_Diastolic'] = st.number_input("BP Diastolic", min_value=50.0, max_value=120.0,
                                               value=float(st.session_state.get('BP_Diastolic', 80.0)))
    user_input['Cholesterol'] = st.number_input("Cholesterol (mg/dL)", min_value=50.0, max_value=400.0,
                                              value=float(st.session_state.get('Cholesterol', 
                                              st.session_state.lab_values.get('Cholesterol', 150) if st.session_state.lab_values else 150)))
    
    stress_options = ["None", "Low", "Medium", "High"]
    user_input['Stress'] = st.selectbox("Stress Level", stress_options, 
                                      index=st.session_state.get('Stress', 0))
    user_input['Hereditary'] = st.radio("Family History of Diabetes", [0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No",
                                      index=st.session_state.get('Hereditary', 0))
    
    st.subheader("Diabetes-specific Tests")
    user_input['HbA1c'] = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, step=0.1,
                                        value=float(st.session_state.get('HbA1c', 
                                        st.session_state.lab_values.get('HbA1c', 5.0) if st.session_state.lab_values else 5.0)))
    user_input['Fasting_Glucose'] = st.number_input("Fasting Glucose (mg/dL)", min_value=50.0, max_value=300.0,
                                                  value=float(st.session_state.get('Fasting_Glucose', 
                                                  st.session_state.lab_values.get('Fasting_Glucose', 90) if st.session_state.lab_values else 90)))
    user_input['Post_Meal_Glucose'] = st.number_input("Post-Meal Glucose (mg/dL)", min_value=50.0, max_value=400.0,
                                                    value=float(st.session_state.get('Post_Meal_Glucose', 
                                                    st.session_state.lab_values.get('Post_Meal_Glucose', 120) if st.session_state.lab_values else 120)))
    user_input['C_Peptide'] = st.number_input("C-Peptide (ng/mL)", min_value=0.1, max_value=8.0, step=0.1,
                                            value=float(st.session_state.get('C_Peptide', 2.0)))
    user_input['Ketones'] = st.number_input("Ketones (mmol/L)", min_value=0.0, max_value=8.0, step=0.1,
                                          value=float(st.session_state.get('Ketones', 0.5)))

# Create user dataframe with ALL features
user_df = pd.DataFrame([{
    'Age': user_input['Age'],
    'Gender': 0 if user_input['Gender'] == "Female" else 1,
    'Height_cm': user_input['Height_cm'],
    'Weight_kg': user_input['Weight_kg'],
    'BMI': user_input['BMI'],
    'BP_Systolic': user_input['BP_Systolic'],
    'BP_Diastolic': user_input['BP_Diastolic'],
    'Cholesterol': user_input['Cholesterol'],
    'Stress': stress_options.index(user_input['Stress']),
    'Hereditary': user_input['Hereditary'],
    'HbA1c': user_input['HbA1c'],
    'Fasting_Glucose': user_input['Fasting_Glucose'],
    'Post_Meal_Glucose': user_input['Post_Meal_Glucose'],
    'C_Peptide': user_input['C_Peptide'],
    'Ketones': user_input['Ketones']
}])

provided_glucose_params = []
if user_input['HbA1c'] > 0:
    provided_glucose_params.append('HbA1c')
if user_input['Fasting_Glucose'] > 0:
    provided_glucose_params.append('Fasting_Glucose')
if user_input['Post_Meal_Glucose'] > 0:
    provided_glucose_params.append('Post_Meal_Glucose')

if len(provided_glucose_params) >= 2:
    with st.container():
        st.subheader("🔍 Preliminary Health Snapshot")
        risk_score = 0
        max_possible_score = 0
        assessment = []
        
        # HbA1c assessment
        if 'HbA1c' in provided_glucose_params:
            hba1c_val = user_input['HbA1c']
            max_possible_score += 30
            if hba1c_val > 6.4:
                assessment.append(("🚨", f"HbA1c ({hba1c_val}%)", "Diabetes range (>6.4%)", "red"))
                risk_score += 30
            elif hba1c_val > 5.6:
                assessment.append(("⚠️", f"HbA1c ({hba1c_val}%)", "Pre-diabetes (5.7-6.4%)", "orange"))
                risk_score += 20
            else:
                assessment.append(("✅", f"HbA1c ({hba1c_val}%)", "Normal (≤5.6%)", "green"))
        
        # Fasting Glucose assessment
        if 'Fasting_Glucose' in provided_glucose_params:
            fasting_val = user_input['Fasting_Glucose']
            max_possible_score += 30
            if fasting_val > 125:
                assessment.append(("🚨", f"Fasting Glucose ({fasting_val} mg/dL)", "Diabetes range (>125 mg/dL)", "red"))
                risk_score += 30
            elif fasting_val > 99:
                assessment.append(("⚠️", f"Fasting Glucose ({fasting_val} mg/dL)", "Pre-diabetes (100-125 mg/dL)", "orange"))
                risk_score += 20
            else:
                assessment.append(("✅", f"Fasting Glucose ({fasting_val} mg/dL)", "Normal (≤99 mg/dL)", "green"))
        
        # Post-Meal Glucose assessment
        if 'Post_Meal_Glucose' in provided_glucose_params:
            post_val = user_input['Post_Meal_Glucose']
            max_possible_score += 20
            if post_val > 200:
                assessment.append(("🚨", f"Post-Meal Glucose ({post_val} mg/dL)", "Diabetes range (>200 mg/dL)", "red"))
                risk_score += 20
            elif post_val > 140:
                assessment.append(("⚠️", f"Post-Meal Glucose ({post_val} mg/dL)", "Pre-diabetes (140-199 mg/dL)", "orange"))
                risk_score += 10
            else:
                assessment.append(("✅", f"Post-Meal Glucose ({post_val} mg/dL)", "Normal (≤140 mg/dL)", "green"))
        
        # Cholesterol assessment
        chol_val = user_input['Cholesterol']
        max_possible_score += 10
        if chol_val > 240:
            assessment.append(("⚠️", f"Cholesterol ({chol_val} mg/dL)", "High (>240 mg/dL)", "orange"))
            risk_score += 10
        elif chol_val > 200:
            assessment.append(("ℹ️", f"Cholesterol ({chol_val} mg/dL)", "Borderline high (200-239 mg/dL)", "blue"))
            risk_score += 5
        else:
            assessment.append(("✅", f"Cholesterol ({chol_val} mg/dL)", "Normal (<200 mg/dL)", "green"))
        
        # BMI assessment
        bmi_val = user_input['BMI']
        max_possible_score += 10
        if bmi_val > 30:
            assessment.append(("🚨", f"BMI ({bmi_val})", "Obesity (>30)", "red"))
            risk_score += 10
        elif bmi_val > 25:
            assessment.append(("⚠️", f"BMI ({bmi_val})", "Overweight (25-30)", "orange"))
            risk_score += 5
        else:
            assessment.append(("✅", f"BMI ({bmi_val})", "Normal (18.5-25)", "green"))
        
        # C-Peptide assessment
        c_peptide_val = user_input['C_Peptide']
        max_possible_score += 10
        if c_peptide_val < 1.0:
            assessment.append(("🟡", f"C-Peptide ({c_peptide_val} ng/mL)", "Low (may indicate Type 1)", "orange"))
            risk_score += 8
        elif c_peptide_val > 4.5:
            assessment.append(("🟡", f"C-Peptide ({c_peptide_val} ng/mL)", "High (may indicate Type 2)", "orange"))
            risk_score += 6
        else:
            assessment.append(("✅", f"C-Peptide ({c_peptide_val} ng/mL)", "Normal (1.1-4.4 ng/mL)", "green"))

        # Ketones assessment  
        ketones_val = user_input['Ketones']
        max_possible_score += 10
        if ketones_val > 3.0:
            assessment.append(("🚨", f"Ketones ({ketones_val} mmol/L)", "High (risk of ketoacidosis)", "red"))
            risk_score += 10
        elif ketones_val > 0.6:
            assessment.append(("⚠️", f"Ketones ({ketones_val} mmol/L)", "Elevated (may indicate Type 1)", "orange"))
            risk_score += 7
        else:
            assessment.append(("✅", f"Ketones ({ketones_val} mmol/L)", "Normal (0.0-0.6 mmol/L)", "green"))

        # Blood Pressure assessment
        bp_sys = user_input['BP_Systolic']
        bp_dia = user_input['BP_Diastolic']
        max_possible_score += 10
        if bp_sys > 140 or bp_dia > 90:
            assessment.append(("⚠️", f"Blood Pressure ({bp_sys}/{bp_dia})", "High (hypertension)", "orange"))
            risk_score += 8
        elif bp_sys > 120 or bp_dia > 80:
            assessment.append(("ℹ️", f"Blood Pressure ({bp_sys}/{bp_dia})", "Elevated (pre-hypertension)", "blue"))
            risk_score += 4
        else:
            assessment.append(("✅", f"Blood Pressure ({bp_sys}/{bp_dia})", "Normal", "green"))
        
        if max_possible_score > 0:
            risk_percentage = min(100, int((risk_score / max_possible_score) * 100))
        else:
            risk_percentage = 0
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Preliminary Risk Score", f"{risk_percentage}%")
        with col2:
            st.progress(risk_percentage / 100)
            
            if risk_percentage > 70:
                st.warning("High diabetes risk indicated")
            elif risk_percentage > 40:
                st.info("Moderate diabetes risk indicated")
            else:
                st.success("Low diabetes risk indicated")
        
        st.markdown("**Health Parameter Analysis:**")
        for item in assessment:
            emoji, param, desc, color = item
            bg_color = {
                "red": "#ffcccc",
                "orange": "#fff3cd",
                "green": "#d4edda",
                "blue": "#e2f0fd"
            }.get(color, "#ffffff")
            
            st.markdown(
                f"""<div style='background-color:{bg_color}; padding:10px; border-radius:5px; margin:5px 0;'>
                {emoji} <b>{param}</b> - {desc}
                </div>""", 
                unsafe_allow_html=True
            )
        
        st.warning("ℹ️ This preliminary assessment is based on available values only. For full AI-powered prediction with all parameters, click 'Predict Diabetes'.")

def get_risk_level(confidence):
    if confidence < 0.6:
        return "🟢 Low"
    elif confidence < 0.8:
        return "🟠 Medium"
    else:
        return "🔴 High"

def single_model_predict(user_df):
    """Get prediction from the single Random Forest model"""
    try:
        # Prepare features in correct order
        X_user = user_df[model_features]
        
        # Get prediction from the model (NO SCALING for Random Forest)
        prediction = model.predict(X_user)[0]
        proba = model.predict_proba(X_user)[0]
        confidence = max(proba)
        
        return prediction, proba, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

if st.button("🔍 Predict Diabetes", type="primary"):
    with st.spinner("Analyzing your data..."):
        try:
            # Use the single model prediction function
            prediction, proba, confidence = single_model_predict(user_df)
            
            if prediction is not None:
                # Store results in session state
                st.session_state.prediction = prediction
                st.session_state.proba = proba
                st.session_state.confidence = confidence
                st.session_state.prediction_made = True
                
                st.subheader("🎯 Prediction Result")
                if prediction == "Diabetes":
                    st.error("⚠️ Diabetes likely detected")
                elif prediction == "Pre-Diabetes":
                    st.warning("🔸 Pre-Diabetes detected")
                elif prediction == "Type 1":
                    st.error("⚠️ Type 1 Diabetes likely detected")
                elif prediction == "Type 2":
                    st.error("⚠️ Type 2 Diabetes likely detected")
                else:
                    st.success("✅ No diabetes detected")
            
            # Display confidence and risk level
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Confidence Level", f"{confidence*100:.1f}%")
            with col2:
                bar_length = int(confidence * 20)
                emoji_bar = "🟥" * bar_length + "⬜" * (20 - bar_length)
                risk_grade = "🟢 Low" if confidence < 0.6 else "🟠 Medium" if confidence < 0.8 else "🔴 High"
                st.metric("🩺 Risk Level", risk_grade)
                st.write(emoji_bar)
                st.progress(float(confidence))
           
            # Display probabilities
            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": [p*100 for p in proba]
            })
            st.markdown("**Class Probabilities:**")
            st.dataframe(prob_df.style.format({"Probability": "{:.2f}%"}), hide_index=True)
            
            # Show feature importance
            st.subheader("🧠 Explanation - Top Influencing Features")
            try:
                # Use Random Forest feature importance
                importances = model.feature_importances_
                
                # Create feature importance dataframe
                feature_imp_df = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                # Plot
                plt.figure(figsize=(10, 6))
                bars = plt.barh(feature_imp_df['Feature'][::-1], feature_imp_df['Importance'][::-1], color='skyblue')
                plt.xlabel('Importance Score')
                plt.title('Top 10 Most Influential Features')
                plt.gca().invert_yaxis()
                
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(plt)
                plt.close()
                
                # Display as table
                st.markdown("**Feature Importance Rankings:**")
                st.dataframe(feature_imp_df.style.format({"Importance": "{:.4f}"}), hide_index=True)
                
                # Store for PDF
                st.session_state.shap_df = feature_imp_df
                st.session_state.shap_fig = plt.gcf()
                
            except Exception as e:
                st.warning(f"Could not generate feature importance: {str(e)}")

            # Save to history
            patient_name = st.session_state.get("report_name", "")
            if patient_name and st.session_state.prediction_made:
                if not st.session_state.patient_id:
                    st.session_state.patient_id = generate_patient_id(patient_name)

                record = {
                    "Patient_ID": st.session_state.patient_id,
                    "Prediction": st.session_state.prediction,
                    "Confidence": round(st.session_state.confidence * 100, 2),
                    "Risk_Level": get_risk_level(st.session_state.confidence),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Input_Method": "Report Upload" if analysis_choice == "📄 Upload Lab Report" else "Manual Entry"
                }
                record.update(user_input)
                csv_file = "patient_history.csv"
                df = pd.DataFrame([record])
                if os.path.exists(csv_file):
                    df.to_csv(csv_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(csv_file, index=False)
                st.session_state.record_saved = True
                st.success(f"🪪 Patient ID `{st.session_state.patient_id}` saved to history.")
            else:
                st.warning("⚠️ Prediction made but not saved to history - please enter patient name")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.session_state.prediction_made = False


def generate_recommendations(user_input):
    recommendations = []
    if user_input['HbA1c'] > 6.4:
        recommendations.append({
            "type": "diet",
            "priority": "high",
            "tip": "Reduce added sugars and refined carbs",
            "reason": f"Your HbA1c ({user_input['HbA1c']}%) indicates diabetes risk"
        })
    elif user_input['HbA1c'] > 5.6:
        recommendations.append({
            "type": "diet", 
            "priority": "medium",
            "tip": "Choose low-glycemic foods (whole grains, legumes)",
            "reason": f"Your HbA1c ({user_input['HbA1c']}%) is borderline high"
        })
    if user_input['Fasting_Glucose'] > 125:
        recommendations.append({
            "type": "diet",
            "priority": "high",
            "tip": "Avoid sugary drinks and late-night snacks",
            "reason": f"Fasting glucose ({user_input['Fasting_Glucose']} mg/dL) is elevated"
        })
    if user_input['BMI'] > 30:
        recommendations.append({
            "type": "exercise",
            "priority": "high",
            "tip": "Start with 30-min daily walks (brisk pace)",
            "reason": f"Your BMI ({user_input['BMI']}) indicates obesity"
        })
    elif user_input['BMI'] > 25:
        recommendations.append({
            "type": "exercise",
            "priority": "medium",
            "tip": "Aim for 10,000 steps/day",
            "reason": f"Your BMI ({user_input['BMI']}) indicates overweight"
        })
    if user_input['Stress'] == "High":
        recommendations.append({
            "type": "mental",
            "priority": "medium", 
            "tip": "Try 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s)",
            "reason": "You reported high stress levels"
        })
    if user_input['Hereditary'] == 1:
        recommendations.append({
            "type": "medical",
            "priority": "medium",
            "tip": "Schedule annual glucose tolerance test",
            "reason": "Family history of diabetes detected"
        })
    
    # New recommendations for additional features
    if user_input['C_Peptide'] < 1.0:
        recommendations.append({
            "type": "medical",
            "priority": "high",
            "tip": "Consult doctor for Type 1 diabetes evaluation",
            "reason": f"Low C-Peptide ({user_input['C_Peptide']} ng/mL) may indicate Type 1 diabetes"
        })
    elif user_input['C_Peptide'] > 4.5:
        recommendations.append({
            "type": "medical",
            "priority": "medium",
            "tip": "Discuss insulin resistance with your doctor",
            "reason": f"High C-Peptide ({user_input['C_Peptide']} ng/mL) may indicate insulin resistance"
        })
    
    if user_input['Ketones'] > 3.0:
        recommendations.append({
            "type": "medical",
            "priority": "high",
            "tip": "Seek immediate medical attention - risk of ketoacidosis",
            "reason": f"High ketones ({user_input['Ketones']} mmol/L) can be dangerous"
        })
    elif user_input['Ketones'] > 0.6:
        recommendations.append({
            "type": "diet",
            "priority": "medium",
            "tip": "Increase carbohydrate intake and stay hydrated",
            "reason": f"Elevated ketones ({user_input['Ketones']} mmol/L) detected"
        })
    
    if user_input['BP_Systolic'] > 140 or user_input['BP_Diastolic'] > 90:
        recommendations.append({
            "type": "medical",
            "priority": "high",
            "tip": "Monitor blood pressure regularly and consult doctor",
            "reason": f"High blood pressure ({user_input['BP_Systolic']}/{user_input['BP_Diastolic']}) increases diabetes complications risk"
        })
    
    return recommendations

def generate_meal_plan(user_input):
    meals = []
    reasons = []
    if user_input['HbA1c'] > 5.6:
        reasons.append("Low-glycemic meals to stabilize blood sugar")
        for meal_type in ["breakfast", "lunch", "dinner"]:
            template = random.choice(MEAL_TEMPLATES[meal_type])
            meal = template.format(
                protein=random.choice(LOW_GI_FOODS["proteins"]),
                veggies=random.choice(LOW_GI_FOODS["veggies"]),
                nuts=random.choice(["almonds", "walnuts"]),
                berries=random.choice(["blueberries", "raspberries"])
            )
            meals.append({"type": meal_type, "meal": meal})
    elif user_input['BMI'] > 25:
        reasons.append("Portion-controlled meals for weight management")
        meals.extend([
            {"type": "breakfast", "meal": "2 boiled eggs + 1 slice whole grain toast"},
            {"type": "lunch", "meal": "Grilled chicken salad (1 cup greens + 3oz chicken)"},
            {"type": "dinner", "meal": "4oz salmon + 1/2 cup quinoa + steamed broccoli"}
        ])
    else:
        reasons.append("Balanced meals to maintain health")
        meals.extend([
            {"type": "breakfast", "meal": "Smoothie (spinach, banana, almond milk)"},
            {"type": "lunch", "meal": "Turkey and avocado whole wheat sandwich"},
            {"type": "dinner", "meal": "Lentil curry with brown rice"}
        ])
    return {"meals": meals, "reasons": reasons}

def generate_workout_plan(user_input):
    plan = {}
    if user_input['BMI'] > 30 or user_input['Age'] > 60:
        level = "beginner"
    else:
        level = "intermediate"
    if user_input['Fasting_Glucose'] > 100:
        plan["cardio"] = {
            "workouts": random.sample(WORKOUT_PLANS[level]["cardio"], 2),
            "reason": f"Your glucose levels ({user_input['Fasting_Glucose']} mg/dL) benefit from cardio"
        }
    if user_input['BMI'] > 25:
        plan["strength"] = {
            "workouts": random.sample(WORKOUT_PLANS[level]["strength"], 2),
            "reason": f"Your BMI ({user_input['BMI']}) responds well to strength training"
        }
    if user_input['Stress'] == "High":
        plan["tip"] = EXERCISE_TIPS["high_stress"]
    return plan

def generate_mental_health_tips(user_input):
    tips = []
    if user_input['Stress'] == "High":
        tips.append({
            "type": "stress",
            "activities": random.sample(MENTAL_ACTIVITIES["high_stress"], 2),
            "reason": "Your reported high stress levels increase diabetes risk"
        })
    elif user_input['Stress'] in ["Medium", "Low"]:
        tips.append({
            "type": "stress",
            "activities": random.sample(MENTAL_ACTIVITIES["moderate_stress"], 1),
            "reason": "Regular stress management prevents blood sugar spikes"
        })
    if user_input['BMI'] > 25 or user_input['Stress'] == "High":
        tips.append({
            "type": "sleep",
            "tips": random.sample(SLEEP_TIPS["poor_sleep"], 2),
            "reason": "Quality sleep helps regulate hunger hormones"
        })
    
    return tips

def generate_medical_advice(user_input):
    actions = []
    if user_input['HbA1c'] > 6.4:
        actions.append({
            "type": "urgent",
            "items": MEDICAL_CHECKS["high_hba1c"],
            "reason": f"Your HbA1c ({user_input['HbA1c']}%) indicates diabetes risk"
        })
    elif user_input['HbA1c'] > 5.6:
        actions.append({
            "type": "preventive",
            "items": random.sample(MEDICAL_CHECKS["high_hba1c"], 1) + [
                "Discuss lifestyle change programs"
            ],
            "reason": f"Borderline HbA1c ({user_input['HbA1c']}%) detected"
        })
    if user_input['BMI'] > 30:
        actions.append({
            "type": "urgent",
            "items": MEDICAL_CHECKS["high_bmi"],
            "reason": f"Your BMI ({user_input['BMI']}) requires evaluation"
        })
    if user_input['Hereditary'] == 1:
        actions.append({
            "type": "preventive",
            "items": MEDICAL_CHECKS["family_history"],
            "reason": "Family history increases your risk"
        })
    
    return actions





def generate_summary(user_input, prediction, confidence):
    summary_parts = []
    
    hba1c = float(user_input['HbA1c'])
    if hba1c > 6.4:
        summary_parts.append("elevated HbA1c levels")
    elif hba1c > 5.6:
        summary_parts.append("borderline HbA1c levels")
        
    fasting = float(user_input['Fasting_Glucose'])
    if fasting > 125:
        summary_parts.append("high fasting glucose")
    elif fasting > 99:
        summary_parts.append("elevated fasting glucose")
        
    post_meal = float(user_input['Post_Meal_Glucose'])
    if post_meal > 200:
        summary_parts.append("very high post-meal glucose")
    elif post_meal > 140:
        summary_parts.append("elevated post-meal glucose")
        
    bmi = float(user_input['BMI'])
    if bmi > 30:
        summary_parts.append("obesity (BMI > 30)")
    elif bmi > 25:
        summary_parts.append("overweight (BMI > 25)")
        
    if int(user_input['Hereditary']):
        summary_parts.append("family history of diabetes")
    
    age = int(user_input['Age'])
    if age > 45:
        summary_parts.append("age over 45")
    if prediction == "No Diabetes" and confidence < 0.6 and not summary_parts:
        return "✅ Your values show no diabetes indicators, and the AI model predicts low risk."
    if prediction == "No Diabetes" and summary_parts:
        return (
            "⚠️ The model predicts no diabetes, but your inputs show some risk factors like "
            + ", ".join(summary_parts[:-1])
            + (" and " if len(summary_parts) > 1 else "")
            + summary_parts[-1]
            + ". Consider lifestyle changes to reduce risk."
        )
    if prediction == "Pre-Diabetes":
        return (
            "🔸 The model predicts pre-diabetes. Your inputs show "
            + ", ".join(summary_parts[:-1])
            + (" and " if len(summary_parts) > 1 else "")
            + summary_parts[-1]
            + ". Consult a doctor for preventive measures."
        )
    if prediction == "Diabetes":
        return (
            "🚨 The model predicts diabetes. Your inputs indicate "
            + ", ".join(summary_parts[:-1])
            + (" and " if len(summary_parts) > 1 else "")
            + summary_parts[-1]
            + ". Please consult a doctor immediately for diagnosis and treatment."
        )
        
    return "🩺 No clear risk detected. Maintain healthy habits."

st.subheader("📝 AI Summary")
if st.session_state.prediction_made:
    summary_text = generate_summary(user_input, st.session_state.prediction, st.session_state.confidence)
    st.markdown(summary_text)

if st.session_state.prediction_made:
    st.markdown("---")
    if st.button("🌱 Show Personalized Recommendations", key="show_recommendations"):
        st.subheader("🌱 Your Personalized Recommendations")
        recommendations = generate_recommendations(user_input)
        if not recommendations:
            st.success("🎉 All your health markers are in optimal ranges! Keep up the good work!")
        else:
            recommendations.sort(key=lambda x: x["priority"], reverse=True)
            for category in ["diet", "exercise", "mental", "medical"]:
                category_tips = [r for r in recommendations if r["type"] == category]
                if category_tips:
                    with st.expander(f"{category.title()} Tips", expanded=True):
                        for tip in category_tips:
                            st.warning(f"🔹 **{tip['tip']}**")  
                            st.caption(f"*Why? {tip['reason']}*")
                            st.write("") 

        st.markdown("---")
        st.subheader("🍽️ Personalized Meal Plan")
        meal_plan = generate_meal_plan(user_input)
        tab1, tab2 = st.tabs(["Meals", "Nutrition Tips"])
        with tab1:
            for meal in meal_plan["meals"]:
                with st.container(border=True):
                    st.markdown(f"**{meal['type'].title()}**")
                    st.write(meal["meal"])
                    st.progress(70 if meal["type"]=="dinner" else 50)  
        with tab2:
            st.write("**Key Goals:**")
            for reason in meal_plan["reasons"]:
                st.write(f"- {reason}")
            st.markdown("""
            **Smart Eating Tips:**
            - Chew slowly to improve digestion
            - Drink water before meals
            - Use smaller plates for portion control
            """)
        st.markdown("---")
        st.subheader("💪 Your Exercise Plan")
        workout_plan = generate_workout_plan(user_input)
        if not workout_plan:
            st.info("All your exercise markers are optimal! Keep being active!")
        else:
            tab1, tab2 = st.tabs(["Workouts", "Science"])
            with tab1:
                if "cardio" in workout_plan:
                    with st.expander("🏃 Cardio", expanded=True):
                        for workout in workout_plan["cardio"]["workouts"]:
                            st.write(f"- {workout}")
                
                if "strength" in workout_plan:
                    with st.expander("🏋️ Strength", expanded=True):
                        for workout in workout_plan["strength"]["workouts"]:
                            st.write(f"- {workout}")
            with tab2:
                st.write("**Why This Works:**")
                if "cardio" in workout_plan:
                    st.caption(workout_plan["cardio"]["reason"])
                if "strength" in workout_plan:
                    st.caption(workout_plan["strength"]["reason"])
                if "tip" in workout_plan:
                    st.success(f"💡 Bonus Tip: {workout_plan['tip']}")
        st.markdown("---")
        st.subheader("🧘 Mental Wellness Tips")
        mental_tips = generate_mental_health_tips(user_input)
        if not mental_tips:
            st.success("🌟 Your stress levels are well-managed! Keep practicing mindfulness!")
        else:
            for tip in mental_tips:
                with st.expander(f"{'Stress Reduction' if tip['type'] == 'stress' else 'Sleep Improvement'}", expanded=True):
                    st.write(f"**Try these:**")
                    for item in tip["activities"] if tip["type"] == "stress" else tip["tips"]:
                        st.write(f"- {item}")
                    st.caption(f"*Why? {tip['reason']}*")

        st.markdown("---")
        st.subheader("🩺 Medical Action Plan")
        medical_advice = generate_medical_advice(user_input)
        if not medical_advice:
            st.success("🎉 No urgent medical actions needed! Maintain regular checkups.")
        else:
            tab1, tab2 = st.tabs(["Tests/Screenings", "Doctor Questions"])
            with tab1:
                for advice in medical_advice:
                    st.warning(f"**{advice['reason']}**")
                    for item in advice["items"]:
                        st.write(f"- {item}")
            with tab2:
                if user_input['HbA1c'] > 5.6:
                    st.write("**Ask your doctor:**")
                    for q in DOCTOR_QUESTIONS["prediabetes"]:
                        st.write(f"- '{q}'")
                
                if user_input['Cholesterol'] > 200:
                    st.write("**Cholesterol questions:**")
                    for q in DOCTOR_QUESTIONS["high_cholesterol"]:
                        st.write(f"- '{q}'")
st.subheader("📄 Generate PDF Report")
patient_name = st.text_input("Patient Name", key="report_name")

if patient_name and st.session_state.prediction_made:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Diabetes Risk Assessment Report")
    c.line(50, height - 60, width - 50, height - 60)
    
    # Patient Info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Patient: {patient_name}")
    c.drawString(50, height - 110, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(50, height - 130, f"Input Method: {'Report Upload' if analysis_choice == '📄 Upload Lab Report' else 'Manual Entry'}")
    
    # Prediction Results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 160, f"Prediction: {st.session_state.prediction}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 180, f"Confidence: {st.session_state.confidence*100:.1f}%")
    risk = get_risk_level(st.session_state.confidence)
    c.drawString(50, height - 200, f"Risk Level: {risk}")

    # Health Metrics
    y_pos = height - 230
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Health Metrics:")
    c.setFont("Helvetica", 12)
    y_pos -= 20
    for key, val in user_input.items():
        c.drawString(70, y_pos, f"{key}: {val}")
        y_pos -= 15
        if y_pos < 150:
            c.showPage()
            y_pos = height - 100

    # Lab Values (if available)
    if st.session_state.lab_values:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Extracted Lab Values:")
        c.setFont("Helvetica", 12)
        y_pos -= 20
        for k, v in st.session_state.lab_values.items():
            c.drawString(70, y_pos, f"{k}: {v}")
            y_pos -= 15
            if y_pos < 100:
                c.showPage()
                y_pos = height - 100

    # Feature Importance
    if st.session_state.shap_df is not None:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos - 10, "Top Influencing Features:")
        c.setFont("Helvetica", 12)
        y_pos -= 30
        for i, row in st.session_state.shap_df.head(3).iterrows():
            c.drawString(70, y_pos, f"{i+1}. {row['Feature']} (Impact: {row['Impact']:.2f})")
            y_pos -= 20
        y_pos -= 10

    # AI Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "AI Summary:")
    y_pos -= 20
    c.setFont("Helvetica", 12)
    summary_lines = wrap(summary_text, width=100)
    for line in summary_lines:
        c.drawString(70, y_pos, line)
        y_pos -= 15
        if y_pos < 100:
            c.showPage()
            y_pos = height - 100

    # Feature Importance Visualization
    if st.session_state.shap_fig:
        try:
            img_buf = io.BytesIO()
            st.session_state.shap_fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            img = ImageReader(img_buf)
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "📊 Feature Importance Visualization")
            img_width = 480  
            img_height = 350
            x_pos = 50
            y_pos = height - 420  
            c.drawImage(img, x_pos, y_pos, width=img_width, height=img_height, 
                        preserveAspectRatio=True, mask='auto')
            y_pos = height - 450  # Reset position after image
        except Exception as e:
            st.warning(f"Could not embed feature importance image in PDF: {e}")

    # Recommendations Header (New Page)
    c.showPage()
    y_pos = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_pos, "🌟 Personalized Recommendations")
    y_pos -= 30

    # Get all recommendations
    all_recommendations = {
        "Diet": generate_meal_plan(user_input),
        "Exercise": generate_workout_plan(user_input),
        "Mental Health": generate_mental_health_tips(user_input),
        "Medical": generate_medical_advice(user_input)
    }

    # Add recommendations to PDF
    for category, data in all_recommendations.items():
        if data:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, f"{category}:")
            y_pos -= 20
            c.setFont("Helvetica", 12)
            
            if category == "Diet":
                for meal in data["meals"]:
                    c.drawString(70, y_pos, f"- {meal['type'].title()}: {meal['meal']}")
                    y_pos -= 15
            elif category == "Exercise":
                if "cardio" in data:
                    c.drawString(70, y_pos, "Cardio:")
                    y_pos -= 15
                    for workout in data["cardio"]["workouts"]:
                        c.drawString(90, y_pos, f"› {workout}")
                        y_pos -= 15
                if "strength" in data:
                    c.drawString(70, y_pos, "Strength:")
                    y_pos -= 15
                    for workout in data["strength"]["workouts"]:
                        c.drawString(90, y_pos, f"› {workout}")
                        y_pos -= 15
            elif category == "Mental Health":
                for tip in data:
                    c.drawString(70, y_pos, f"- {tip['activities'][0] if tip['type']=='stress' else tip['tips'][0]}")
                    y_pos -= 15
            elif category == "Medical":
                for action in data:
                    c.drawString(70, y_pos, f"- {action['items'][0]}")
                    y_pos -= 15
            
            y_pos -= 10
            if y_pos < 100:
                c.showPage()
                y_pos = height - 50

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 30, "Note: This report is AI-generated and should be reviewed by a medical professional.")
    c.save()
    buffer.seek(0)

    st.download_button(
        "⬇️ Download Full Report",
        buffer,
        file_name=f"{patient_name}_Diabetes_Report.pdf",
        mime="application/pdf"
    )

    st.subheader("📧 Email Report")
    col1, col2 = st.columns(2)
    with col1:
        sender_email = st.text_input("Your Email")
        sender_password = st.text_input("App Password", type="password")
    with col2:
        receiver_email = st.text_input("Recipient Email")
    
    if st.button("📤 Send Email") and sender_email and sender_password and receiver_email:
        try:
            msg = EmailMessage()
            msg['Subject'] = f"Diabetes Report - {patient_name}"
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg.set_content(f"""
                Hi,
                
                Please find attached the diabetes risk assessment report for {patient_name}.
                
                Regards,
                Diabetes Predictor AI
            """)
            msg.add_attachment(
                buffer.getvalue(),
                maintype='application',
                subtype='pdf',
                filename=f"{patient_name.replace(' ', '_')}_Diabetes_Report.pdf"
            )
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(sender_email, sender_password)
                smtp.send_message(msg)  

            st.success("📤 Email sent successfully!")
            
        except Exception as e:
            st.error(f"❌ Email failed: {e}")
else:
    st.warning("⚠️ Please fill all email fields (sender, password, receiver) to send the report.")

st.markdown("---")
st.subheader("📋 View Patient History")  
try:
    if os.path.exists("patient_history.csv"):
        try:
            history_df = pd.read_csv("patient_history.csv")
            with st.expander("📜 Show Patient History"):
                st.dataframe(history_df, use_container_width=True)
                
                with st.form("clear_form"):
                    st.warning("⚠️ This will permanently delete all saved history!")
                    confirm_clear = st.checkbox("Yes, delete all history")
                    if st.form_submit_button("🗑️ Clear History") and confirm_clear:
                        os.remove("patient_history.csv")
                        st.success("✅ All patient history cleared.")
                        time.sleep(1)  
                        st.rerun()
        except Exception as e:
            st.error(f"Error reading history file: {str(e)}")
            pd.DataFrame(columns=[
                'Patient_ID', 'Prediction', 'Confidence', 'Risk_Level', 'Timestamp',
                'Input_Method', 'Age', 'Gender', 'BMI', 'Cholesterol', 'Stress',
                'Hereditary', 'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose'
            ]).to_csv("patient_history.csv", index=False)
            st.info("Created a new empty history file.")
    else:
        st.info("No patient history found. Make predictions to build history.")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")