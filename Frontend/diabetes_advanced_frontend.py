
import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import StandardScaler
import re
import pdfplumber
import pytesseract
from PIL import Image
import re
import database
import json
import plotly.express as px
import sqlite3
from database import get_db_connection, generate_unique_user_id, get_user_by_id, get_user_by_email, create_user
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import smtplib
from email.message import EmailMessage
from textwrap import wrap
from datetime import datetime
import openai
import os
from typing import Dict, List

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



def clean_extracted_name(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()

    # Ignore if it contains address/lab/doctor markers
    blacklist = [
        "road", "colony", "hospital", "clinic", "survey",
        "barcode", "lic", "kelkar", "labs", "healthcare", "dr."
    ]
    if any(b.lower() in name.lower() for b in blacklist):
        return None

    # Ignore if it's too short or looks like a code
    if len(name.split()) < 2:
        return None

    return name



def extract_values_from_report(file):
    """Process uploaded reports using OCR/text extraction"""
    text = ""
    
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
    
    return text

def is_valid_test_name(test_name):
    """Filter out invalid test names that are actually reference ranges, sample types, etc."""
    invalid_patterns = [
        r'reference', r'range', r'sample', r'method', r'note', r'interpretation',
        r'normal', r'abnormal', r'high', r'low', r'equal', r'guideline', r'clinical',
        r'biological', r'interval', r'collected', r'received', r'reported',
        r'fasting plasma glucose equal', r'impaired fasting glucose',
        r'impaired glucose', r'diabetes mellitus', r'random plasma glucose',
        r'kindly correlate', r'test done on', r'non-diabetic', r'pre-diabetic',
        r'diabetic', r'good control', r'fair control', r'poor control'
    ]
    
    test_name_lower = test_name.lower()
    
    # Skip if it contains any invalid patterns
    if any(pattern in test_name_lower for pattern in invalid_patterns):
        return False
    
    # Skip if it's too short or doesn't look like a test name
    if len(test_name) < 3 or test_name.isdigit():
        return False
    
    # Skip if it's just a unit or method
    if test_name_lower in ['mg/dl', 'mg', '%', 'mmol/l', 'plasma', 'serum', 'blood', 'urine']:
        return False
    
    return True

def extract_lab_address(text):
    """Extract lab address information from text"""
    address_patterns = [
        # Pattern for footer addresses
        r'(?:Office|Address|Location)[:\s]*([^\n]{10,150}?(?:Pune|Road|Colony|Center)[^\n]{0,50})',
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[,\s]+(?:Pune|Hadapsar|Kothrud)[^\n]{10,100}',
        r'(?:Opp|Near|Behind).*?[A-Z][a-z]+.*?\d{6}',
        r'\b\d{3,6}\b.*?(?:Road|Street|Avenue|Lane).*?(?:Pune|City)',
        # Specific patterns from your reports
        r'Office No[.:]\s*\d+[^\n]{10,100}4110\d{2}',
        r'[A-Z][a-z]+\s+Center[^\n]{10,100}4110\d{2}',
    ]
    
    addresses = []
    for pattern in address_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            address = match.strip(' :,-')
            if len(address) > 15 and '4110' in address:  # Pune area code
                addresses.append(address)
    
    # Return the most likely address (longest one usually)
    if addresses:
        return max(addresses, key=len)
    return None

def extract_lab_contact(text):
    """Extract lab contact information"""
    contact_patterns = [
        r'Tel[:\s]*([+\d\s-]+)',
        r'Phone[:\s]*([+\d\s-]+)',
        r'Contact[:\s]*([+\d\s-]+)',
    ]
    
    for pattern in contact_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def clean_extracted_lab_name(lab_name: str) -> str | None:
    if not lab_name:
        return None
        
    # Initial cleaning
    lab_name = lab_name.strip(' :,-*•\t\r\n')
    
    # Remove common non-lab text that might appear in headers
    lab_name = re.sub(r'(?:dr\.|doctor|md|mbbs|dnb|sample|collected|received|reported|regards|technologist|page|end of report|interpretation|guidelines|reference|interval|test result|unit|biological|normal|patient|name|age|sex|gender|barcode|ref|printed|report released|sent by|direct)', '', lab_name, flags=re.IGNORECASE)
    lab_name = lab_name.strip()
    
    # Remove page numbers and other numeric artifacts
    lab_name = re.sub(r'\bPage\s*\d+\s*of\s*\d+\b', '', lab_name, flags=re.IGNORECASE)
    lab_name = re.sub(r'\b\d+\s*/\s*\d+\b', '', lab_name)
    lab_name = lab_name.strip()
    
    # Check if it looks like a person's name (usually 2-3 words, title case)
    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', lab_name):
        return None
        
    # Check if it's just initials or codes
    if re.match(r'^[A-Z]{1,3}(?:\s+[A-Z]{1,3})*$', lab_name):
        return None
        
    # Check length requirements
    if len(lab_name) < 3 or len(lab_name) > 100:
        return None
        
    return lab_name


def extract_lab_name_by_visual_heuristics(text):
    """
    Extract lab name based on visual heuristics - looks for prominent text
    that appears at the top of pages with different formatting
    """
    # Split text into lines and analyze each line
    lines = text.split('\n')
    lab_name_candidates = []
    
    # Heuristic 1: Look for lines that are likely headers (first few lines)
    header_lines = lines[:10]  # First 10 lines of each page
    
    for line in header_lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are clearly not lab names
        if is_likely_not_lab_name(line):
            continue
            
        # Score based on visual presentation clues
        score = 0
        
        # All caps suggests header/title
        if line.isupper():
            score += 3
            
        # Contains lab-related keywords
        lab_keywords = ['lab', 'laboratory', 'diagnostic', 'pathology', 'healthcare', 
                       'services', 'ltd', 'limited', 'inc', 'center', 'centre']
        if any(keyword in line.lower() for keyword in lab_keywords):
            score += 2
            
        # Reasonable length for a lab name (3-50 chars)
        if 3 <= len(line) <= 50:
            score += 1
            
        # Doesn't contain numbers (usually not in lab names)
        if not re.search(r'\d', line):
            score += 1
            
        # Doesn't look like a date
        if not re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
            score += 1
            
        # Doesn't look like a person's name (Mr./Mrs./Dr.)
        if not re.search(r'^(?:Mr|Mrs|Ms|Dr|Miss)\.?\s+[A-Z]', line, re.IGNORECASE):
            score += 1
            
        if score >= 4:  # Good candidate
            lab_name_candidates.append((line, score))
    
    # Heuristic 2: Look for text that appears in similar positions across pages
    if not lab_name_candidates:
        # Look for repeated patterns in first lines of pages
        page_starts = []
        current_page = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Assume page break when we see certain patterns
            if (re.search(r'page\s*\d+\s*of\s*\d+', line.lower()) or 
                re.search(r'^\d+\s*/\s*\d+$', line) or
                len(current_page) > 50):  # Arbitrary page length
                if current_page:
                    page_starts.append(current_page[:5])  # First 5 lines of each page
                current_page = []
            current_page.append(line)
        
        # Find common patterns in page starts
        common_candidates = []
        for page_start in page_starts:
            for line in page_start:
                if not is_likely_not_lab_name(line):
                    common_candidates.append(line)
        
        # Count occurrences and take the most common
        from collections import Counter
        if common_candidates:
            counter = Counter(common_candidates)
            most_common = counter.most_common(3)
            for candidate, count in most_common:
                if count >= 2:  # Appears on multiple pages
                    lab_name_candidates.append((candidate, 5))  # High confidence
    
    # Return the best candidate
    if lab_name_candidates:
        # Sort by score and return the highest
        lab_name_candidates.sort(key=lambda x: x[1], reverse=True)
        return lab_name_candidates[0][0]
    
    return None

def is_likely_not_lab_name(text):
    """Check if text is likely NOT a lab name"""
    text_lower = text.lower()
    
    # Blacklist patterns
    blacklist = [
        # Dates and numbers
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates
        r'page\s*\d+\s*of\s*\d+',          # Page numbers
        r'^\d+$',                          # Just numbers
        r'\d{2}:\d{2}',                    # Time
        r'reg no',                         # Registration numbers
        r'pid',                            # Patient IDs
        r'vid',                            # Visit IDs
        
        # Personal names and titles
        r'^(?:mr|mrs|ms|dr|miss)\.?\s+',
        r'patient', r'name', r'age', r'sex', r'gender',
        
        # Technical terms
        r'sample', r'collected', r'received', r'reported',
        r'reference', r'interval', r'normal', r'abnormal',
        
        # Address components (but not complete addresses)
        r'road', r'street', r'avenue', r'colony', r'near', r'opp',
        r'pune', r'mumbai', r'city',  # These might be in lab names, but careful
        
        # Too short or too long
        r'^.{0,2}$',                    # Too short
        r'^.{100,}$',                   # Too long (probably address)
    ]
    
    for pattern in blacklist:
        if re.search(pattern, text_lower):
            return True
    
    # Specific case: "Sanjana Vivek Kelkar" looks like a person's name
    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$', text):
        return True
        
    # Contains mostly special characters
    if re.match(r'^[^a-zA-Z0-9\s]{3,}$', text):
        return True
        
    return False

def extract_prominent_header(text):
    """
    Extract the most prominent header text (likely lab name)
    by analyzing text that appears in consistent positions
    """
    lines = text.split('\n')
    
    # Strategy 1: Look for text that appears in the first 3 lines of logical sections
    section_starts = []
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_section:
                section_starts.append(current_section[:3])  # First 3 lines of each section
                current_section = []
            continue
        current_section.append(line)
    
    # Collect candidate lines from section starts
    candidates = []
    for section in section_starts:
        for line in section:
            if (len(line) > 5 and 
                not is_likely_not_lab_name(line) and
                not line.isdigit()):
                candidates.append(line)
    
    # Strategy 2: Look for ALL CAPS lines that aren't too long
    all_caps_candidates = []
    for line in lines:
        line = line.strip()
        if (line.isupper() and 
            10 <= len(line) <= 60 and 
            not is_likely_not_lab_name(line)):
            all_caps_candidates.append(line)
    
    # Combine and prioritize
    all_candidates = candidates + all_caps_candidates
    
    if all_candidates:
        # Count occurrences and take most frequent
        from collections import Counter
        counter = Counter(all_candidates)
        most_common = counter.most_common(5)
        
        for candidate, count in most_common:
            # Additional validation
            if (not re.search(r'\d', candidate) and  # No numbers
                not re.search(r'[.:]$', candidate) and  # Doesn't end with punctuation
                len(candidate.split()) >= 1 and  # At least one word
                len(candidate.split()) <= 5):    # Not too many words
                return candidate
    
    return None

def extract_all_values(text):
    """
    SUPERCHARGED extraction - finds EVERYTHING in the report
    """
    # First, clean the text thoroughly
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', text)  # Remove spaces between letters and numbers
    text = re.sub(r'(\d)\s+([a-zA-Z%])', r'\1\2', text)  # Remove spaces between numbers and letters/%
    
    extracted_data = {
        'demographics': {},
        'tests': {},
        'lab_info': {}
    }
    
    # ===== DEMOGRAPHICS EXTRACTION =====
    # Name patterns (comprehensive for all formats) - KEEP YOUR WORKING VERSION
    
    
    name_patterns = [
    r"(?i)^(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s*([A-Z]+\s+[A-Z]+)",
   
   


    
    r"(?im)^\s*Patient\s*Name\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*(?:$|\r|\n)(?=\s*Barcode\s*[:\-])",
    r"(?im)^\s*Patient\s*Name\s*[:\-]?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)\s*(?:$|\r|\n)(?=\s*Barcode\s*[:\-])",
    r"(?im)Patient\s*Name\s*[:\-]?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*(?:Age\/Gender|Age\s*[:\-]|Gender|Sex))(?!Barcode|ID|No)",
    r"(?im)^\s*(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*(?=Age\s*[:\-])",
    r"(?im)^\s*(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.)?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)\s*(?=Age\s*[:\-])",
  




    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)",

    # Name before Age/Gender (for Mr. RAHUL JAIL type)
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*Age\s*[:\-])",
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.)?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)(?=\s*Age\s*[:\-])",

    # Restrict fallback to only match if "Patient", "Name", or "Mr/Mrs" present
    r"(?i)(?:(?:Patient\s*Name)|(?:Name)|(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.))\s*[:\-]?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*(?:Age|Gender|Sex))",
    # Pattern for first report format: "Mr. RAHUL JAIL" followed by details
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)(?=\s*(?:PUNE|Tel No|PIN No|PID NO|Age))",
    
    # Pattern for second report format: "Patient Name : Seema Avinash Garud"
    r"(?i)Patient\s*Name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    
    # Pattern for table format with colon
    r"(?i)Patient Name\s*[:\-]\s*([A-Za-z\s]+?)(?=\s*(?:Age|Gender|Sex|Order|$))",
    
    # Pattern for all caps names before Age/Sex
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)(?=\s*Age\s*[:\-])",
    
    # Pattern for mixed case names before Age/Sex
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?=\s*Age\s*[:\-])",
    
    # Fallback: Look for name-like patterns near demographic info
    r"(?i)(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*(?:\d{2}Y|Tel No|PIN No))",
    
    # Your original patterns with corrected flag placement
    r"(?i)(?:Patient\s*Name|Name)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*(?=Age|Gender|Sex|Order|$)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)\s*(?=Age|Gender|Sex)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*(?=Age|Gender|Sex)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Za-z][A-Za-z\s]+?)\s*(?=Age\s*\/Gender)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Z\s]+[A-Z])(?=\s*Age\s*[:\-])",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Z\s]+[A-Z])(?=\s*Sex\s*[:\-])",
    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+?)(?=\s*Age\/Gender)",
    r"(?i)Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+?)(?=\s*Age\/Gender)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Za-z\s]+?)(?=\s*Age\s*[\/:]\s*\d+)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Za-z\s]+?)(?=\s*Age\/Gender)",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*Patient\s*Name\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender|$))",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*Name\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender|$))",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*Patient\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender))",
    r"(?i)(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*(?:Age|Sex|Gender))",
    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender|$))",
    r"(?i)Name\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender|$))",
    r"(?i)Patient\s*[:\-]?\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender))",
    r"(?i)([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})(?=\s*(?:Age|Sex|Gender))",
    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s*(?:Age|Gender|Sex|Order|$))",
    r"(?i)Name\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s*(?:Age|Gender|Sex|Order|$))",
    r"(?i)Patient\s*[:\-]?\s*([A-Za-z\s]+)(?=\s*(?:Age|Gender|Sex|Order))",
    r"(?i)Name\s*[:\-]\s*([A-Z][A-Za-z\s]+?)(?=\s*(?:Age|Sex|Gender|$))",
    r"(?i)Patient Name[\s|]*([A-Z][A-Za-z\s]+?)(?=[\s|]*(?:Age|Sex|Gender))",
    r"(?i)Patient\s*Name\s*[:\-]?\s*([A-Za-z\s]+)(?=\s*(?:Age|Gender|Sex|Order|$|\n))",
    r"(?i)Name\s*[:\-]?\s*([A-Za-z\s]+)(?=\s*(?:Age|Gender|Sex|Order|$|\n))",
    r"(?i)Patient\s*Name\s*[:\-]?\s*(.*?)(?=\s*Age\/Gender)",
    r"(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=\s*Age\/Gender)",
    

    ]

    
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip(' :,-')
            # Clean up name (remove titles, extra spaces)
            name = re.sub(r'^\s*(?:Mr|Ms|Mrs|Dr)\.?\s*', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name)
            extracted_data['demographics']['Name'] = name.title()
            break
    
    # Age patterns (comprehensive for all formats) - KEEP YOUR WORKING VERSION
    age_patterns = [
    r"(?i)Age\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*Years?",
    r"(?i)Age\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*Yrs?",
    r"(?i)Age\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*Yr",
    r"(?i)Age\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*Y\b",
    r"(?i)Age/Gender\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)",
    r"(?i)\((\d{1,3}(?:\.\d{1,2})?)\)\s*Years?",
    r"(?i)Age\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\b(?=\s*(?:Sex|Gender))"
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                age_value = float(match.group(1))
                extracted_data['demographics']['Age'] = int(age_value)
                break
            except:
                continue
    
    # Gender patterns (comprehensive for all formats) - KEEP YOUR WORKING VERSION
    gender_patterns = [
    r"(?i)Sex\s*[:\-]?\s*(Male|Female|M|F)\b",
    r"(?i)Gender\s*[:\-]?\s*(Male|Female|M|F)\b",
    r"(?i)Sex/Gender\s*[:\-]?\s*(Male|Female|M|F)\b",
    r"(?i)Age.*?(Male|Female|M|F)\b"
    ]
    
    for pattern in gender_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gender_str = match.group(1).upper()
            if gender_str.startswith('M') or gender_str == 'M':
                extracted_data['demographics']['Sex'] = "Male"
            elif gender_str.startswith('F') or gender_str == 'F':
                extracted_data['demographics']['Sex'] = "Female"
            break
    
        # ===== LAB INFO EXTRACTION =====
    extracted_data['lab_info'] = {}
    
    # 1. First try visual heuristics for lab name
    lab_name = extract_lab_name_by_visual_heuristics(text)
    if not lab_name:
        lab_name = extract_prominent_header(text)
    
    if lab_name:
        extracted_data['lab_info']['Lab_Name'] = lab_name
    
    # 2. Then try the other patterns for additional info
    lab_name_patterns = [
        r"(?im)^\s*([A-Z][A-Za-z\s&.,-]{5,50}?(?:LABS|LABORATORIES|DIAGNOSTIC|PATHOLOGY|HEALTHCARE|SERVICES|LTD|LIMITED))",
        r"(?im)^\s*([A-Z][A-Za-z\s&.,-]{5,40})\s*$",
        r"(?im)^\s*([A-Z][A-Za-z\s&.,-]{5,40})\s*Laboratory Diagnostic Services",
    ]
    
    for pattern in lab_name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate != lab_name:  # Don't duplicate
                extracted_data['lab_info']['Lab_Name_Alt'] = candidate
            break
        # Extract Doctor/Pathologist name separately
    doctor_patterns = [
        r"(?i)dr\.\s+([A-Z][a-z]+\s+[A-Z][a-zA-Z.\s]+)(?=\s*(?:MD|MBBS|DNB|Biochemistry|Pathology|MMC Reg\. No))",
        r"(?i)Regards,[,\s]*([A-Za-z\s]+)(?=\s*Lab Technologist)",
    ]
    
    for pattern in doctor_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            doctor_name = match.group(1).strip()
            # Clean up the doctor name
            doctor_name = re.sub(r'(?:MD|MBBS|DNB|MMC Reg\. No\.?)[\s\d\/]*', '', doctor_name, flags=re.IGNORECASE)
            doctor_name = doctor_name.strip()
            if doctor_name and len(doctor_name) > 5:
                extracted_data['lab_info']['Pathologist'] = doctor_name
            break

    # 2. LAB ADDRESS - Look for address patterns (this is often missing in OCR text)
    # 2. LAB ADDRESS - Look for address patterns
    address_patterns = [
    r"(?i)Address[:\s]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    r"(?i)Location[:\s]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    r"(?i)Colony[:\s]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    r"(?i)Road[:\s]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    r"(?i)Office[:\s]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    # Extract from processing location which often contains address
    r"(?i)Processing Location[:\s-]*([^\n]{10,100}?)(?=\s*(?:Tel|Phone|Contact|$))",
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            address = match.group(1).strip()
            if len(address) > 10:  # Ensure it's a reasonable address length
                extracted_data['lab_info']['Lab_Address'] = address
                break

    # 3. LAB CONTACT - Extract contact information
    contact_patterns = [
        r'Contact[:\s-]*([+\d\s-]{8,15})',
        r'Tel[:\s-]*([+\d\s-]{8,15})',
        r'Phone[:\s-]*([+\d\s-]{8,15})',
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\d{5}[-.\s]?\d{5}\b'
    ]
    
    for pattern in contact_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_data['lab_info']['Lab_Contact'] = match.group(1).strip()
            break

    # 4. LAB ID/ACCESSION NUMBER - Very important for reports
    lab_id_patterns = [
        r"(?i)LAB ID\s*:\s*(\d+)",
        r"(?i)Report No\s*:\s*(\w+)",
        r"(?i)Accession No\s*:\s*(\w+)",
    ]
    
    for pattern in lab_id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_data['lab_info']['Lab_ID'] = match.group(1).strip()
            break

    # 5. COLLECTION POINT - Extract where sample was collected
    collection_patterns = [
        r"(?i)Collected At[:\s]*([^\n]{5,50}?)(?=\s*,\s*Received At)",
        r"(?i)Sample Collection[:\s]*([^\n]{5,50}?)(?=\s*\d{2}:\d{2})",
    ]
    
    for pattern in collection_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            collection_point = match.group(1).strip()
            # Clean up any remaining timestamps
            collection_point = re.sub(r'\d{2}/\d{2}/\d{4}\s*\d{2}:\d{2}', '', collection_point).strip()
            if collection_point and len(collection_point) > 3:
                extracted_data['lab_info']['Collection_Point'] = collection_point
            break
        # Test date - Enhanced Patterns
    date_patterns = [
        r"(?i)(?:Report\s*Date|Date\s*of\s*Report)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?i)(?:Collection\s*Date|Date\s*of\s*Collection|Sample\s*Collected\s*On)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?i)Date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?i)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?=\s*(?:Report|Collection|Sample))",
        # Formats like "dd-mm-yyyy", "dd/mm/yyyy", "mm/dd/yyyy"
        r"(?i)(\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b)",
        # Formats like "yyyy-mm-dd"
        r"(?i)(\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])\b)",
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_data['lab_info']['Test_Date'] = match.group(1).strip()
            break
    
    # ===== FILTER FUNCTION =====
    def is_valid_test_name(test_name):
        """Filter out invalid test names that are actually reference ranges, sample types, etc."""
        invalid_patterns = [
            r'reference', r'range', r'sample', r'method', r'note', r'interpretation',
            r'normal', r'abnormal', r'high', r'low', r'equal', r'guideline', r'clinical',
            r'biological', r'interval', r'collected', r'received', r'reported',
            r'fasting plasma glucose equal', r'impaired fasting glucose',
            r'impaired glucose', r'diabetes mellitus', r'random plasma glucose',
            r'kindly correlate', r'test done on', r'non-diabetic', r'pre-diabetic',
            r'diabetic', r'good control', r'fair control', r'poor control'
        ]
        
        test_name_lower = test_name.lower()
        
        # Skip if it contains any invalid patterns
        if any(pattern in test_name_lower for pattern in invalid_patterns):
            return False
        
        # Skip if it's too short or doesn't look like a test name
        if len(test_name) < 3 or test_name.isdigit():
            return False
        
        # Skip if it's just a unit or method
        if test_name_lower in ['mg/dl', 'mg', '%', 'mmol/l', 'plasma', 'serum', 'blood', 'urine']:
            return False
        
        return True
    
    # ===== COMPREHENSIVE TEST EXTRACTION =====
    all_tests = {}
    
    # Master pattern to find ALL test values - IMPROVED VERSION
    master_patterns = [
        # Pattern: Test Name: Value Unit (with better filtering)
        r'([A-Z][A-Za-z\s/&-]+(?:GLUCOSE|SUGAR|HbA1c|A1c|CHOLESTEROL|CHOL|HDL|LDL|TRIG))[:\s]+([\d.]+)\s*(mg/dl|mg/dL|%|mmol/L)',
        
        # Pattern: Value Unit (Test Name) - more specific
        r'([\d.]+)\s*(mg/dl|mg/dL|%|mmol/L)\s*[\(]?([A-Z][A-Za-z\s/&]+(?:GLUCOSE|SUGAR|HbA1c))[\)]?',
        
        # Pattern for table format: Test Name | Value | Unit
        r'([A-Z][A-Za-z\s/&-]+(?:GLUCOSE|SUGAR|HbA1c))[\s|:]*([\d.]+)\s*[\|]?\s*(mg/dl|mg/dL|%)',
    ]

    for pattern in master_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) == 3:
                if match[0].replace('.', '').isdigit():  # Value Unit TestName pattern
                    try:
                        value = float(match[0])
                        unit = match[1]
                        test_name = match[2].strip()
                    except ValueError:
                        continue
                else:  # TestName: Value Unit pattern
                    test_name = match[0].strip()
                    try:
                        value = float(match[1])
                        unit = match[2]
                    except ValueError:
                        continue
                
                # Clean up test name - remove trailing colons, dashes, etc.
                test_name = re.sub(r'[\s:|()-]+$', '', test_name)
                test_name = re.sub(r'^[\s:|()-]+', '', test_name)
                
                # CRITICAL: Use the filter function to exclude invalid test names
                if not is_valid_test_name(test_name):
                    continue
                    
                # Only add if it looks like a valid medical test
                if (len(test_name) > 2 and not test_name.lower() in ['page', 'reference', 'range'] 
                    and not re.search(r'\b(?:or|and|the|a|an|of|for)\b', test_name.lower())):
                    all_tests[test_name] = {'value': value, 'unit': unit}
    
    # ===== TABLE FORMAT EXTRACTION =====
    def extract_from_table_format(text):
        """Extract values from table-formatted lab reports"""
        table_tests = {}
        
        # Pattern for: Test Name | Value | Unit | Reference Range
        table_pattern = r'([A-Z][A-Za-z\s/-]+)[\s|:]*([\d.]+)\s*[\|]?\s*(mg/dl|%|mmol/L)[\s|]*(?:[\d.-]+\s*(?:mg/dl|%))?'
        
        matches = re.findall(table_pattern, text)
        for match in matches:
            if len(match) == 3:
                test_name = match[0].strip()
                try:
                    value = float(match[1])
                    unit = match[2]
                    
                    if is_valid_test_name(test_name):
                        table_tests[test_name] = {'value': value, 'unit': unit}
                except ValueError:
                    continue
        
        return table_tests

    # Extract from table format and add to all_tests
    table_tests = extract_from_table_format(text)
    all_tests.update(table_tests)
    
    # ===== IMPROVED SPECIFIC PATTERNS =====
    specific_patterns = {
        'HbA1c': [
            r'GLYCATED HAEMOGLOBIN.*?([\d.]+)\s*%',
            r'HbA1c.*?([\d.]+)\s*%',
            r'A1c.*?([\d.]+)\s*%',
            r'Glycated Hemoglobin.*?([\d.]+)\s*%',
            r'Glycosylated Hemoglobin.*?([\d.]+)\s*%',
            r"(?:HbA1c|Hb\s*A1c|HBA1C|HgbA1c|Hgb\s*A1c|Glycated Hemoglobin|Glycosylated Hemoglobin)[^\d]{0,10}([\d.]+)\s*%",
            r"([\d.]+)\s*%\s*(?:HbA1c|HBA1C|HgbA1c|Glycated Hemoglobin)",
            r"Non[- ]diabetic.*?([\d.]+)\s*%"
        ],
        'Fasting Glucose': [
            r'Glucose,\s*Fasting[^\d]{0,20}([\d.]+)\s*mg/dl',
            r'Glucose\s*Fasting[^\d]{0,20}([\d.]+)\s*mg/dl',
            r'Fasting Blood Sugar[^\d]{0,20}([\d.]+)\s*mg/dl',
            r'FASTING PLASMA GLUCOSE.*?([\d.]+)\s*mg/dl',
            r'Fasting Plasma Glucose.*?([\d.]+)\s*mg/dl',
            r'Fasting Glucose.*?([\d.]+)\s*mg/dl',
            r'FBS.*?([\d.]+)\s*mg/dl',
            r'FBG.*?([\d.]+)\s*mg/dl',
            r"(?:Fasting Plasma Glucose|Fasting Blood Glucose|FBS|FBG|FPG|Pre[- ]?Breakfast Glucose|Basal Blood Glucose|Glucose,\s*Fasting|Glucose\s*Fasting)[^\d]{0,10}([\d.]+)\s*mg/dl",
            r"([\d.]+)\s*mg/dl.*?(?:Fasting Plasma Glucose|Fasting Blood Glucose|FBS|FBG|FPG|Glucose,\s*Fasting|Glucose\s*Fasting)"
            r"Glucose,\s*Fasting[^\d]{0,10}([\d.]+)\s*mg/dl",
            r"Glucose\s*Fasting[^\d]{0,10}([\d.]+)\s*mg/dl"
        ],
        'Postprandial Glucose': [
            r'POST-PRANDIAL PLASMA GLUCOSE.*?([\d.]+)\s*mg/dl',
            r'Post-Prandial Plasma Glucose.*?([\d.]+)\s*mg/dl',
            r'POST PRANDIAL GLUCOSE.*?([\d.]+)\s*mg/dl',
            r'PPBS.*?([\d.]+)\s*mg/dl',
            r'PPG.*?([\d.]+)\s*mg/dl',
            r'2 hr.*?GLUCOSE.*?([\d.]+)\s*mg/dl',
            r'2 hour.*?GLUCOSE.*?([\d.]+)\s*mg/dl',
            r"(?:Postprandial Glucose|Post[- ]Meal Glucose|Post Lunch Glucose|PPBS|PPBG|PPG|PLBS|PP2BS|2\s*hr\s*PP|2\s*Hour\s*PP|After Meal Glucose)[^\d]{0,10}([\d.]+)\s*mg/dl",
            r"([\d.]+)\s*mg/dl.*?(?:Postprandial Glucose|Post[- ]Meal Glucose|PPBS|PPBG|PPG|2\s*hr\s*PP|PP2BS)"
        ]
    }

    # Process each pattern type
    for test_name, patterns in specific_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    value = float(match.group(1))
                    unit = 'mg/dL' if 'Glucose' in test_name else '%'
                    
                    # Add to all_tests with consistent naming
                    all_tests[test_name] = {'value': value, 'unit': unit}
                    break  # Stop after first successful match
                except (ValueError, IndexError):
                    continue
    
    # ===== SPECIAL HbA1c EXTRACTION =====
    # Special handling for HbA1c that appears after "Non-diabetic" text
    hba1c_alternative_patterns = [
        r'Non-diabetic.*?([\d.]+)\s*%',
        r'GLYCATED HAEMOGLOBIN.*?([\d.]+)\s*%\s*Non-diabetic',
        r'HbA1c.*?:\s*([\d.]+)\s*%\s*\(.*?Non-diabetic\)'
    ]

    for pattern in hba1c_alternative_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and 'HbA1c' not in all_tests:
            try:
                value = float(match.group(1))
                all_tests['HbA1c'] = {'value': value, 'unit': '%'}
                break
            except (ValueError, IndexError):
                continue
    
    # ===== SPECIAL POST-MEAL EXTRACTION =====
    # Special handling for Post-Meal glucose variations
    post_meal_alternative_patterns = [
        r'POST.*?MEAL.*?GLUCOSE.*?([\d.]+)\s*mg/dl',
        r'PLASMA GLUCOSE.*?POST.*?([\d.]+)\s*mg/dl',
        r'Glucose.*?Post.*?([\d.]+)\s*mg/dl',
        r'PP.*?([\d.]+)\s*mg/dl.*?(?:meal|prandial)',
        r'2\s*hr.*?([\d.]+)\s*mg/dl'
    ]

    for pattern in post_meal_alternative_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and 'Postprandial Glucose' not in all_tests:
            try:
                value = float(match.group(1))
                all_tests['Postprandial Glucose'] = {'value': value, 'unit': 'mg/dL'}
                break
            except (ValueError, IndexError):
                continue

    # After extracting all tests, add this validation
    if 'Fasting Glucose' in all_tests and 'Postprandial Glucose' in all_tests:
        fasting_value = all_tests['Fasting Glucose']['value']
        post_meal_value = all_tests['Postprandial Glucose']['value']
    
    # If post-meal value is suspiciously similar to fasting value
        if abs(fasting_value - post_meal_value) < 10:  # Within 10 mg/dL difference
        # Likely error - remove the post-meal value
            del all_tests['Postprandial Glucose']
    # After all extraction, check if post-meal is missing
    if 'Postprandial Glucose' not in all_tests:
    # Look for any indication that post-meal test was done but value is missing
        post_meal_mentioned = re.search(r'POST.*?MEAL|PPBS|2\s*hr', text, re.IGNORECASE)
        if post_meal_mentioned:
            print("DEBUG: Post-meal test mentioned but value not found")
        # You can set a default or leave it as missing

    eag_patterns = [
    r'Estimated Average Glucose.*?([\d.]+)\s*mg/dl',
    r'Average Estimated Glucose.*?([\d.]+)\s*mg/dl',
    r'eAG.*?([\d.]+)\s*mg/dl',
    ]

    for pattern in eag_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                all_tests['Estimated Average Glucose'] = {'value': value, 'unit': 'mg/dL'}
            # Remove if it was mistakenly added as glucose
                if 'Postprandial Glucose' in all_tests and all_tests['Postprandial Glucose']['value'] == value:
                    del all_tests['Postprandial Glucose']
                break
            except (ValueError, IndexError):
                continue
    
    # ===== FINAL CLEANUP =====
    # Remove any tests that might have slipped through the filters
    final_tests = {}
    for test_name, test_data in all_tests.items():
        if is_valid_test_name(test_name):
            final_tests[test_name] = test_data
    
    extracted_data['tests'] = final_tests
    
    return extracted_data
# Load your models and components
try:
    stage_model = joblib.load('diabetes_stage_model1.pkl')
    stage_scaler = joblib.load('stage_scaler1.pkl')
    stage_le = joblib.load('stage_label_encoder1.pkl')
    type_model = joblib.load('diabetes_type_model1.pkl')
    type_scaler = joblib.load('type_scaler1.pkl')
    type_le = joblib.load('type_label_encoder1.pkl')
    feature_columns = joblib.load('feature_columns1.pkl')
    model_metadata = joblib.load('model_metadata1.pkl')
except FileNotFoundError:
    st.error("❌ Model files not found. Please make sure you've trained the models first.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()


def show_landing_page():
    """
    Displays the landing page with options for new and returning users.
    Handles the entire authentication flow.
    """
    if st.session_state.logged_in:
        display_logout_button()
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("---")
    
    # Initialize critical session state variables if they don't exist
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # If user is already logged in, skip the landing page
    if st.session_state.logged_in:
        # Set the app to start from the first input page
        if 'current_page' not in st.session_state or st.session_state.current_page == 0:
            st.session_state.current_page = 1
        st.rerun()
        return

    # Main landing page content
    st.header("Welcome! Please identify yourself to continue.")

    # Use CSS to ensure both columns have the same height
    st.markdown("""
    <style>
    .equal-height {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .equal-height > div:first-child {
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wrap content in a container with equal height class
        st.markdown('<div class="equal-height">', unsafe_allow_html=True)
        st.subheader("New User")
        st.markdown("""
        You're here for the first time.
        We'll generate a unique ID for you to use on future visits.
        """)
        if st.button("Get Started", key="new_user_btn", use_container_width=True):
            st.session_state.registration_step = "generate_id"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Wrap content in a container with equal height class
        st.markdown('<div class="equal-height">', unsafe_allow_html=True)
        st.subheader("Returning User")
        st.markdown("""
        Already have a User ID?
        Enter it below to access your history.
        """)
        if st.button("I have an ID", key="returning_user_btn", use_container_width=True):
            st.session_state.registration_step = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    if 'registration_step' in st.session_state:
        handle_auth_steps()

def handle_auth_steps():
    """Handles the different steps of the authentication process"""
    conn = get_db_connection()
    
    try:
        # STEP 1: Generate ID for new user
        if st.session_state.registration_step == "generate_id":
            # Clear the main content area and only show the auth process
            st.empty()  # Clear any previous content
            
            st.info("🔐 Generating your unique User ID...")
            new_user_id = generate_unique_user_id(conn)
            st.session_state.temp_user_id = new_user_id
            st.session_state.registration_step = "register_details"
            st.rerun()
        
        # STEP 2: Collect details for new user
        elif st.session_state.registration_step == "register_details":
            # Clear the main content area and only show the auth process
            st.empty()  # Clear any previous content
            
            st.success(f"Your unique User ID is: `{st.session_state.temp_user_id}`")
            st.warning("**⚠️ Please save this ID for future logins!**")
            
            # Create a form for registration
            with st.form("registration_form"):
                full_name = st.text_input("Full Name", placeholder="e.g., Jane Smith",autocomplete="off")
                email = st.text_input("Email Address", placeholder="e.g., jane@example.com",autocomplete="off")
                
                # Add a required field indicator
                st.caption("💡 Both fields are required to complete registration.")
                
                # Create columns for the buttons inside the form
                col1, col2 = st.columns(2)
                
                with col1:
                    # Submit button for registration
                    submitted = st.form_submit_button("Complete Registration", use_container_width=True)
                
                with col2:
                    # Cancel button - this will reset the form but we'll handle the logic outside
                    cancelled = st.form_submit_button("Cancel Registration", use_container_width=True, type="secondary")
                
                if submitted:
                    if full_name.strip() and email.strip():  # Check if not empty after removing spaces
                        # Create user in database
                        create_user(conn, st.session_state.temp_user_id, full_name, email)
                        
                        # Set session state
                        st.session_state.user_id = st.session_state.temp_user_id
                        st.session_state.user_name = full_name
                        st.session_state.user_email = email
                        st.session_state.logged_in = True
                        
                        st.success(f"Welcome, {full_name}! Registration successful.")
                        st.balloons()
                        st.session_state.current_page = 6  # Redirect to dashboard
                        st.rerun()
                        
                       
                    else:
                        st.error("❌ Please fill in all required fields to complete registration.")
            
            # Handle cancellation OUTSIDE the form
            if cancelled:
                st.info("Registration cancelled. Your generated ID will not be saved.")
                # Clear the temporary state completely
                del st.session_state.registration_step
                del st.session_state.temp_user_id
                st.rerun()
        
        # STEP 3: Login for returning user
        elif st.session_state.registration_step == "login":
            # Clear the main content area and only show the auth process
            st.empty()  # Clear any previous content
            
            with st.form("login_form"):
                user_id_input = st.text_input("Enter your User ID",autocomplete="off").strip().upper()
        
                if st.form_submit_button("Login"):
                    user_data = get_user_by_id(conn, user_id_input)
            
                    if user_data:
                        st.session_state.user_id = user_data['user_id']
                        st.session_state.user_name = user_data['full_name']
                        st.session_state.user_email = user_data['email']
                        st.session_state.logged_in = True
                
                        st.success(f"Welcome back, {user_data['full_name']}!")
                
                # NEW: Redirect to dashboard instead of main app
                        st.session_state.current_page = 6  # Dashboard page
                        st.rerun()
                
                    else:
                        st.error("User ID not found. Please check or try the New User option.")
        
        # STEP 4: Recovery for returning user who forgot ID
        elif st.session_state.registration_step == "recover_id":
            # Clear the main content area and only show the auth process
            st.empty()  # Clear any previous content
            
            st.info("Recover your User ID by email")
            
            with st.form("recovery_form"):
                email_input = st.text_input("Enter your registered email",autocomplete="off").strip().lower()
                
                if st.form_submit_button("Recover ID"):
                    user_data = get_user_by_email(conn, email_input)
                    
                    if user_data:
                        st.success(f"Your User ID is: `{user_data['user_id']}`")
                        st.info("You can now go back and login with this ID.")
                    else:
                        st.error("Email not found in our system. Please check or register as a new user.")
    
    finally:
        conn.close()
    
    # Add a recovery option on login page
    if (st.session_state.registration_step == "login" and 
        st.button("Forgot your User ID?")):
        st.session_state.registration_step = "recover_id"
        st.rerun()
    
    # Add a back button for all auth steps
    if st.button("← Back to main menu"):
        if 'registration_step' in st.session_state:
            del st.session_state.registration_step
        if 'temp_user_id' in st.session_state:
            del st.session_state.temp_user_id
        st.rerun()

def display_user_info():
    """Display user info in the top left corner of every page"""
    if st.session_state.logged_in:
        # Use a container at the top of the page instead of fixed positioning
        with st.container():
            col1, col2 = st.columns([1, 5])  # First column for user info, second for main content
            
            with col1:
                st.markdown(f"""
                <div style='
                    padding: 8px 12px; 
                    border-radius: 5px; 
                    border: 1px solid #ddd; 
                    font-size: 16px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    👤 <strong style='font-size: 18px;'>{st.session_state.user_name}</strong><br>
                    🆔 <span style='font-size: 16px;'>{st.session_state.user_id}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # This creates an empty column that will push content down
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
def display_logout_button():
    """Display logout button in the main content area"""
    if st.session_state.logged_in:
        # Create a container at the top right of the main content
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🚪 Logout", key="main_logout_btn"):
                # Clear all user-related session state
                for key in ['user_id', 'user_name', 'user_email', 'logged_in', 'current_page']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# Initialize session state for multi-page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0  # CHANGED FROM 1 TO 0 (Landing Page is now page 0)
if 'show_analysis_page' not in st.session_state:
    st.session_state.show_analysis_page = False

# --- Phase 1: User Authentication Variables ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'registration_step' not in st.session_state:
    st.session_state.registration_step = None
if 'temp_user_id' not in st.session_state:
    st.session_state.temp_user_id = None
if 'user_raw_input' not in st.session_state:
    st.session_state.user_raw_input = {}
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'diabetes_stage' not in st.session_state:
    st.session_state.diabetes_stage = None
if 'diabetes_type' not in st.session_state:
    st.session_state.diabetes_type = None
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False
if 'stage_confidence' not in st.session_state:
    st.session_state.stage_confidence = None
if 'type_confidence' not in st.session_state:
    st.session_state.type_confidence = None
if 'used_clinical_rules' not in st.session_state:
    st.session_state.used_clinical_rules = False
if 'clinical_reason' not in st.session_state:
    st.session_state.clinical_reason = None
if 'full_report_data' not in st.session_state:
    st.session_state.full_report_data = None
if 'extraction_status' not in st.session_state:
    st.session_state.extraction_status = None
if 'extracted_values_processed' not in st.session_state:
    st.session_state.extracted_values_processed = False
if 'record_saved' not in st.session_state:
    st.session_state.record_saved = False


# Initialize individual input fields in session state
input_fields = [
    'age', 'gender', 'height', 'weight', 'bmi', 'bp_sys', 'bp_dia', 'chol',
    'stress', 'hered', 'hba1c', 'fasting', 'post_meal', 'c_pep', 'ketones', 'antibodies'
]

for field in input_fields:
    if field not in st.session_state:
        st.session_state[field] = None

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Sidebar - Visible on all pages
with st.sidebar:
    # Progress indicator
    st.subheader("📋 Your Progress")
    pages = ["Basic Info", "Health Metrics", "Screening Tests", "Results", "Detailed Analysis","Lifestyle Recommendations"]  # ADDED "Detailed Analysis"
    for i, page in enumerate(pages, 1):
        status = "✅" if i < st.session_state.current_page else "➡️" if i == st.session_state.current_page else "⏭️"
        st.write(f"{status} Page {i}: {page}")
    
    st.markdown("---")
    
    # Mini quick instructions
    st.subheader("📌 Quick Steps")
    st.markdown("""
    1️⃣ Enter basic info (Age, Gender, Height, Weight)  
    2️⃣ Add health metrics (BP, Cholesterol, Stress, Family History)  
    3️⃣ Provide any screening tests you have (leave others blank)  
    4️⃣ Click **Predict** → Get result  
    5️⃣ If Diabetes → Enter advanced tests for Type 1/Type 2  
    """)
    st.markdown("---")
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This application uses machine learning to analyze your health parameters 
    and estimate your diabetes status: **Non-Diabetic, Pre-Diabetic, or Diabetic** 
    (with further classification into **Type 1 or Type 2 Diabetes** when applicable).
    """)
    
    st.markdown("---")
    if st.session_state.logged_in:
        if st.button("🚪 Logout"):
        # Clear all user-related session state
            for key in ['user_id', 'user_name', 'user_email', 'logged_in', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.write(f"Logged in as: **{st.session_state.user_name}**")
        st.write(f"User ID: `{st.session_state.user_id}`")
    
    # Full instructions
    with st.expander("📖 Detailed Instructions", expanded=True):
        st.markdown("""
        - Fill in **basic information** you know (Age, Gender, Height, Weight)  
        - Provide **health metrics** you know: BP, Cholesterol, Stress Level, Family History  
        - Enter **any screening test results** you have - leave others blank!  
        - Click **'Predict Diabetes Risk'** to get your result  
        - If Diabetes is detected, provide **additional tests** if available
        """)
    
    st.caption("⚠️ This is an AI-based risk assessment tool and should not replace medical consultation.")

# Helper functions
def apply_clinical_rules(hba1c, fasting_glucose, post_meal_glucose):
    """Apply WHO/ADA clinical guidelines with HbA1c as priority"""
    
    # Convert all to float for comparison (handle None values)
    hba1c_val = float(hba1c) if hba1c is not None else None
    fasting_val = float(fasting_glucose) if fasting_glucose is not None else None
    post_meal_val = float(post_meal_glucose) if post_meal_glucose is not None else None
    
    # Count how many valid tests we have
    valid_tests = sum(1 for x in [hba1c_val, fasting_val, post_meal_val] if x is not None)
    
    # If we have at least one test, apply clinical rules with HbA1c priority
    if valid_tests >= 1:
        # DIABETES diagnosis (any one criterion meets diabetes threshold)
        if (hba1c_val is not None and hba1c_val >= 6.5) or \
           (fasting_val is not None and fasting_val >= 126) or \
           (post_meal_val is not None and post_meal_val >= 200):
            return "Diabetes", 0.95, "Clinical Rule: Meets diabetes criteria (HbA1c ≥ 6.5% OR Fasting ≥ 126 OR Post-Meal ≥ 200)"
        
        # PRE-DIABETES diagnosis (any one criterion meets pre-diabetes threshold)
        elif (hba1c_val is not None and hba1c_val >= 5.7) or \
             (fasting_val is not None and fasting_val >= 100) or \
             (post_meal_val is not None and post_meal_val >= 140):
            return "Pre-Diabetes", 0.90, "Clinical Rule: Meets pre-diabetes criteria (HbA1c 5.7-6.4% OR Fasting 100-125 OR Post-Meal 140-199)"
        
        # NORMAL (all available values are normal)
        else:
            return "No Diabetes", 0.90, "Clinical Rule: All available values are within normal range"
    
    # If no tests provided, let ML model decide
    return None, None, None

def prepare_model_input(user_raw_input, feature_columns):
    """Prepare user input for model prediction with proper missing value handling"""
    # Create dictionaries for values and missing flags
    user_values = {}
    missing_flags = {}
    
    # Define mappings for categorical variables
    gender_mapping = {'Female': 0, 'Male': 1, None: np.nan}
    stress_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, None: np.nan}
    
    # Process each feature
    for feature in feature_columns:
        if feature.endswith('_missing'):
            continue  # Skip missing flags, we'll create them
            
        if feature in user_raw_input:
            value = user_raw_input[feature]
            
            # Handle special categorical conversions
            if feature == 'Gender':
                user_values[feature] = gender_mapping[value]
                missing_flags[f"{feature}_missing"] = 0 if value is not None else 1
            elif feature == 'Stress':
                user_values[feature] = stress_mapping[value]
                missing_flags[f"{feature}_missing"] = 0 if value is not None else 1
            else:
                # For numerical values
                user_values[feature] = value
                missing_flags[f"{feature}_missing"] = 0 if value is not None else 1
        else:
            # Feature not provided, set to NaN
            user_values[feature] = np.nan
            missing_flags[f"{feature}_missing"] = 1
    
    # Combine values and missing flags
    final_input = {**user_values, **missing_flags}
    
    # Create DataFrame with correct column order
    user_df = pd.DataFrame([final_input])
    
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = np.nan if not col.endswith('_missing') else 1
    
    return user_df[feature_columns]

def generate_recommendations(user_raw_input, diabetes_stage, has_advanced_data=False):
    """Generate personalized recommendations based on missing information"""
    recommendations = {
        'critical': [],
        'important': [],
        'supplementary': []
    }
    
    # Only show basic test recommendations if user hasn't started advanced testing
    if not has_advanced_data:
        # CRITICAL: Essential diabetes tests
        if user_raw_input.get('HbA1c') is None:
            recommendations['critical'].append("HbA1c test - crucial for diabetes diagnosis")
        if user_raw_input.get('Fasting_Glucose') is None:
            recommendations['critical'].append("Fasting Glucose test - essential for diabetes screening")
        if user_raw_input.get('Post_Meal_Glucose') is None:
            recommendations['critical'].append("Post-Meal Glucose test - important for comprehensive assessment")
        
        # IMPORTANT: Key risk factors and type differentiation
        if user_raw_input.get('Age') is None:
            recommendations['important'].append("Age - helps assess age-related risk factors")
        
        # Check if BMI is missing AND either height or weight is missing
        bmi_missing = user_raw_input.get('BMI') is None
        height_missing = user_raw_input.get('Height_cm') is None
        weight_missing = user_raw_input.get('Weight_kg') is None
        
        if bmi_missing and (height_missing or weight_missing):
            recommendations['important'].append("BMI calculation - important weight-related risk assessment")
        
        if user_raw_input.get('Hereditary') is None:
            recommendations['important'].append("Family history - helps assess genetic predisposition")
        
        # Check if either BP value is missing
        if user_raw_input.get('BP_Systolic') is None or user_raw_input.get('BP_Diastolic') is None:
            recommendations['important'].append("Blood Pressure - hypertension is a diabetes risk factor")
    
    # For diabetes cases: advanced tests for type differentiation
    if diabetes_stage == "Diabetes":
        # If user has started advanced testing OR we're in post-advanced context
        if has_advanced_data or any([
            user_raw_input.get('C_Peptide') is not None,
            user_raw_input.get('Ketones') is not None, 
            user_raw_input.get('Antibodies') is not None
        ]):
            # Check which advanced tests are still missing
            if user_raw_input.get('C_Peptide') is None:
                recommendations['critical'].append("C-Peptide test - crucial for determining diabetes type (Type 1 vs Type 2)")
            if user_raw_input.get('Ketones') is not None:
                recommendations['important'].append("Ketones test - helps identify Type 1 diabetes risk")
            if user_raw_input.get('Antibodies') is not None:
                recommendations['important'].append("Diabetes Antibodies test - important for Type 1 diabetes detection")
        else:
            # If no advanced data provided yet, only suggest basic diabetes confirmation tests
            recommendations['important'].append("Advanced diabetes tests (C-Peptide, Ketones, Antibodies) - for type differentiation if needed")
    
    # Only show supplementary recommendations if user hasn't started advanced testing
    if not has_advanced_data:
        # SUPPLEMENTARY: Additional context
        if user_raw_input.get('Cholesterol') is None:
            recommendations['supplementary'].append("Cholesterol levels - provides complete cardiovascular health picture")
        if user_raw_input.get('Stress') is None:
            recommendations['supplementary'].append("Stress level - stress can affect glucose metabolism")
        if user_raw_input.get('Gender') is None:
            recommendations['supplementary'].append("Gender - provides complete demographic profile")
    
    return recommendations

def display_recommendations(recommendations):
    """Display recommendations in a structured way"""
    if not any(recommendations.values()):  # If all lists are empty
        st.success("🎉 You've provided all recommended information for a comprehensive assessment!")
        return
    
    st.markdown("---")
    st.subheader("📋 Recommendations for Better Assessment")
    
    # CRITICAL recommendations
    if recommendations['critical']:
        st.error("🔴 **CRITICAL - Strongly Recommended**")
        for rec in recommendations['critical']:
            st.write(f"- {rec}")
        st.write("")
    
    # IMPORTANT recommendations
    if recommendations['important']:
        st.warning("🟡 **IMPORTANT - Recommended**")
        for rec in recommendations['important']:
            st.write(f"- {rec}")
        st.write("")
    
    # SUPPLEMENTARY recommendations
    if recommendations['supplementary']:
        st.info("🔵 **SUPPLEMENTARY - Good to Have**")
        for rec in recommendations['supplementary']:
            st.write(f"- {rec}")

def validate_user_input(user_raw_input):
    """
    Validate if user has provided sufficient information for a meaningful prediction
    Returns: (is_valid, error_message)
    """
    # Count how many fields have actual values (not None)
    provided_fields = sum(1 for value in user_raw_input.values() if value is not None)
    
    # Rule 1: Completely empty form
    if provided_fields == 0:
        return False, "❌ Please provide at least some health information to get a prediction."
    
    # Rule 2: Check if only non-informative fields are filled (just Age and/or Gender)
    non_informative_fields = ['Age', 'Gender', 'Stress']
    informative_fields_provided = any(
        user_raw_input.get(field) is not None 
        for field in user_raw_input 
        if field not in non_informative_fields
    )
    
    if not informative_fields_provided:
        return False, "⚠️ Please provide at least one health metric (Height/Weight, Blood Pressure, Cholesterol, Family History, or any glucose test) for a meaningful prediction."
    
    # Rule 3: Check if we have at least SOME health data
    health_fields = ['Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 
                    'Cholesterol', 'Hereditary', 'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose']
    health_data_provided = any(user_raw_input.get(field) is not None for field in health_fields)
    
    if not health_data_provided:
        return False, "⚠️ Please provide some health information like Height/Weight, Blood Pressure, or any test results for diabetes assessment."
    
    # All validation passed
    return True, "✅ Sufficient information provided for prediction."



def save_inputs_to_session():
    """Save current inputs to session state"""
    # Get all the current values from session state
    st.session_state.user_raw_input.update({
        'Age': st.session_state.get('age', None),
        'Gender': st.session_state.get('gender', None),
        'Height_cm': st.session_state.get('height', None),
        'Weight_kg': st.session_state.get('weight', None),
        'BMI': st.session_state.get('bmi', None),
        'BP_Systolic': st.session_state.get('bp_sys', None),
        'BP_Diastolic': st.session_state.get('bp_dia', None),
        'Cholesterol': st.session_state.get('chol', None),
        'Stress': st.session_state.get('stress', None),
        'Hereditary': st.session_state.get('hered', None),
        'HbA1c': st.session_state.get('hba1c', None),
        'Fasting_Glucose': st.session_state.get('fasting', None),
        'Post_Meal_Glucose': st.session_state.get('post_meal', None),
        'C_Peptide': st.session_state.get('c_pep', None),
        'Ketones': st.session_state.get('ketones', None),
        'Antibodies': st.session_state.get('antibodies', None)
    })



from database import get_user_health_records

def calculate_progress_trends(records):
    """Calculate progress trends between latest and previous records"""
    if len(records) < 2:
        return []  # Need at least 2 records for comparison
    
    trends = []
    
    # Get the two most recent records
    latest_record = records[0]
    previous_record = records[1]
    
    latest_clinical = json.loads(latest_record['clinical_data'])
    previous_clinical = json.loads(previous_record['clinical_data'])
    
    # Define metrics to track with their thresholds
    metrics_to_track = {
        'HbA1c': {'threshold': 0.5, 'unit': '%', 'name': 'HbA1c'},
        'Fasting_Glucose': {'threshold': 10, 'unit': 'mg/dL', 'name': 'Fasting Glucose'},
        'Post_Meal_Glucose': {'threshold': 15, 'unit': 'mg/dL', 'name': 'Post-Meal Glucose'},
        'Cholesterol': {'threshold': 20, 'unit': 'mg/dL', 'name': 'Cholesterol'},
        'BMI': {'threshold': 1.0, 'unit': '', 'name': 'BMI'}
    }
    
    for metric, config in metrics_to_track.items():
        current_val = latest_clinical.get(metric)
        previous_val = previous_clinical.get(metric)
        
        # Only calculate if both values exist and are numeric
        if (current_val is not None and previous_val is not None and 
            isinstance(current_val, (int, float)) and isinstance(previous_val, (int, float))):
            
            difference = current_val - previous_val
            threshold = config['threshold']
            
            if difference < -threshold:  # Improvement
                trend = {
                    'metric': config['name'],
                    'change': abs(difference),
                    'unit': config['unit'],
                    'status': 'improved',
                    'message': f"🎉 Your {config['name']} improved by {abs(difference):.1f}{config['unit']}!",
                    'color': 'green'
                }
                trends.append(trend)
                
            elif difference > threshold:  # Deterioration
                trend = {
                    'metric': config['name'],
                    'change': difference,
                    'unit': config['unit'],
                    'status': 'deteriorated',
                    'message': f"⚠️ Your {config['name']} increased by {difference:.1f}{config['unit']}. Please consult your doctor.",
                    'color': 'red'
                }
                trends.append(trend)
                
            else:  # Stable
                trend = {
                    'metric': config['name'],
                    'change': difference,
                    'unit': config['unit'],
                    'status': 'stable',
                    'message': f"➡️ Your {config['name']} remained stable ({difference:+.1f}{config['unit']}).",
                    'color': 'blue'
                }
                trends.append(trend)
    
    return trends
def show_history_dashboard():
    """
    Displays the user's health history and trends.
    """
    st.title("📊 Your Health History")
    st.markdown("---")
    
    # Get user's records from database
    conn = get_db_connection()
    try:
        # Get all records for this user
        all_records = get_user_health_records(conn, st.session_state.user_id)
        
        if not all_records:
            st.info("You don't have any health records yet. Complete a prediction to see your history here!")
            if st.button("← Back to Main App"):
                st.session_state.current_page = 1
            return
        
        # Display summary
        st.success(f"📈 You have {len(all_records)} health record(s) in your history.")
        
        # Show latest 5 records in a table
        st.subheader("🕐 Recent Records")
        recent_records = all_records[:5]  # Get latest 5 records
        
        # Prepare data for display
        display_data = []
        for record in recent_records:
            clinical_data = json.loads(record['clinical_data'])
            prediction_data = json.loads(record['prediction_results'])
            
            # Calculate age group for display
            age = clinical_data.get('Age')
            age_group = "Unknown"
            if age:
                if age < 18: age_group = "Child"
                elif age < 40: age_group = "Young Adult" 
                elif age < 60: age_group = "Middle Age"
                else: age_group = "Senior"
            
            # Create record data - ADD ADVANCED FIELDS
            record_data = {
                'Date': record['date_submitted'].split()[0],
                'Test Date': record['test_date'] or 'Manual',
                'Age': age,
                'Age Group': age_group,
                'HbA1c': clinical_data.get('HbA1c'),
                'Fasting Glucose': clinical_data.get('Fasting_Glucose'),
                'Post-Meal Glucose': clinical_data.get('Post_Meal_Glucose'),
                'Prediction': prediction_data.get('stage'),
                'Confidence': f"{prediction_data.get('stage_confidence', 0) * 100:.1f}%"
            }
            
            # ADD ADVANCED FIELDS IF THEY EXIST
            advanced_fields = {
                'C-Peptide': clinical_data.get('C_Peptide'),
                'Ketones': clinical_data.get('Ketones'),
                'Antibodies': clinical_data.get('Antibodies')
            }
            
            # Only add advanced fields if they have values
            for field, value in advanced_fields.items():
                if value is not None:
                    record_data[field] = value
            
            display_data.append(record_data)
        
        # Display as a nice table
        df = pd.DataFrame(display_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Test Date": "Test Date",
                "Age": st.column_config.NumberColumn("Age", format="%d"),
                "Age Group": "Age Group",
                "HbA1c": st.column_config.NumberColumn("HbA1c", format="%.1f%%"),
                "Fasting Glucose": st.column_config.NumberColumn("Fasting", format="%d mg/dL"),
                "Post-Meal Glucose": st.column_config.NumberColumn("Post-Meal", format="%d mg/dL"),
                "C-Peptide": st.column_config.NumberColumn("C-Peptide", format="%.1f ng/mL"),
                "Ketones": st.column_config.NumberColumn("Ketones", format="%.1f mmol/L"),
                "Antibodies": "Antibodies",
                "Prediction": "Result",
                "Confidence": "Confidence"
            }
        )
        
        # Add Lab Data Viewing Feature
        st.markdown("---")
        st.subheader("🔬 Lab Report Details")

        # Get lab metadata for records that have it
        lab_records = []
        for record in recent_records:
            if record['input_method'] == 'lab_report':
                lab_data = conn.execute(
                    "SELECT * FROM lab_reports_metadata WHERE record_id = ?",
                    (record['record_id'],)
                ).fetchone()
                
                if lab_data:
                    lab_records.append({
                        'record_date': record['date_submitted'].split()[0],
                        'lab_data': lab_data
                    })

        if lab_records:
            for lab_record in lab_records:
                with st.expander(f"📄 Lab Report from {lab_record['record_date']}", expanded=False):
                    lab_data = lab_record['lab_data']
                    
                    col_lab1, col_lab2 = st.columns(2)
                    
                    with col_lab1:
                        if lab_data['lab_name']:
                            st.write(f"**Lab Name:** {lab_data['lab_name']}")
                        if lab_data['pathologist_name']:
                            st.write(f"**Pathologist:** {lab_data['pathologist_name']}")
                        if lab_data['accession_number']:
                            st.write(f"**Accession #:** {lab_data['accession_number']}")
                    
                    with col_lab2:
                        if lab_data['lab_address']:
                            st.write(f"**Address:** {lab_data['lab_address']}")
                        if lab_data['lab_contact']:
                            st.write(f"**Contact:** {lab_data['lab_contact']}")
                    
                    # Show complete extracted lab data
                    if lab_data['extracted_lab_data']:
                        st.markdown("---")
                        st.write("**Complete Extracted Data:**")
                        complete_data = json.loads(lab_data['extracted_lab_data'])
                        
                        if 'demographics' in complete_data and complete_data['demographics']:
                            with st.expander("👤 Patient Details from Report"):
                                st.json(complete_data['demographics'])
                        
                        if 'tests' in complete_data and complete_data['tests']:
                            with st.expander("🧪 All Tests from Report (Including Irrelevant)"):
                                # Show all tests, not just diabetes-related ones
                                all_tests = []
                                for test_name, test_data in complete_data['tests'].items():
                                    all_tests.append({
                                        'Test': test_name,
                                        'Value': test_data.get('value', 'N/A'),
                                        'Unit': test_data.get('unit', 'N/A')
                                    })
                                
                                if all_tests:
                                    st.dataframe(pd.DataFrame(all_tests), use_container_width=True)
                                else:
                                    st.info("No test data extracted")
        else:
            st.info("No lab reports found in your recent records.")
        
        if len(all_records) >= 2:
            st.markdown("---")
            st.subheader("📊 Your Progress Summary")
            
            trends = calculate_progress_trends(all_records)
            
            if trends:
                # Create columns for better layout
                cols = st.columns(2)
                col_index = 0
                
                for trend in trends:
                    with cols[col_index]:
                        if trend['status'] == 'improved':
                            st.success(trend['message'])
                        elif trend['status'] == 'deteriorated':
                            st.error(trend['message'])
                        else:
                            st.info(trend['message'])
                    
                    col_index = (col_index + 1) % 2  # Alternate between columns
            else:
                st.info("No significant changes detected in your key health metrics.")
        # Show trends if we have enough data
        if len(all_records) >= 2:
            st.markdown("---")
            st.subheader("📈 Trends Over Time")
            
            # Prepare data for trends
            trend_data = []
            for record in all_records:
                clinical_data = json.loads(record['clinical_data'])
                trend_data.append({
                    'date': record['date_submitted'],
                    'HbA1c': clinical_data.get('HbA1c'),
                    'Fasting_Glucose': clinical_data.get('Fasting_Glucose'),
                    'Post_Meal_Glucose': clinical_data.get('Post_Meal_Glucose')
                })
            
            trend_df = pd.DataFrame(trend_data)
            trend_df['date'] = pd.to_datetime(trend_df['date'])
            
            # Create tabs for different metrics
            tab1, tab2, tab3 = st.tabs(["HbA1c Trend", "Fasting Glucose Trend", "Post-Meal Glucose Trend"])
            
            with tab1:
                if trend_df['HbA1c'].notna().sum() >= 2:
                    fig = px.line(trend_df.dropna(subset=['HbA1c']), 
                                 x='date', y='HbA1c',
                                 title='HbA1c Trend Over Time',
                                 markers=True)
                    fig.update_layout(yaxis_title="HbA1c (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add trend analysis
                    latest_hba1c = trend_df['HbA1c'].dropna().iloc[0]
                    if latest_hba1c > 6.4:
                        st.error("🚨 Your latest HbA1c indicates diabetes range.")
                    elif latest_hba1c > 5.6:
                        st.warning("⚠️ Your latest HbA1c indicates pre-diabetes range.")
                    else:
                        st.success("✅ Your latest HbA1c is in normal range.")
                else:
                    st.info("Not enough HbA1c data to show trends.")
            
            with tab2:
                if trend_df['Fasting_Glucose'].notna().sum() >= 2:
                    fig = px.line(trend_df.dropna(subset=['Fasting_Glucose']), 
                                 x='date', y='Fasting_Glucose',
                                 title='Fasting Glucose Trend Over Time',
                                 markers=True)
                    fig.update_layout(yaxis_title="Glucose (mg/dL)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough Fasting Glucose data to show trends.")
            
            with tab3:
                if trend_df['Post_Meal_Glucose'].notna().sum() >= 2:
                    fig = px.line(trend_df.dropna(subset=['Post_Meal_Glucose']), 
                                 x='date', y='Post_Meal_Glucose',
                                 title='Post-Meal Glucose Trend Over Time',
                                 markers=True)
                    fig.update_layout(yaxis_title="Glucose (mg/dL)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough Post-Meal Glucose data to show trends.")
        
        # Export functionality
                # Export functionality
        st.markdown("---")
        st.subheader("📤 Export Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Full History (CSV)", use_container_width=True):
                # Convert all records to CSV
                export_data = []
                for record in all_records:
                    clinical_data = json.loads(record['clinical_data'])
                    prediction_data = json.loads(record['prediction_results'])
                    
                    export_record = {
                        'Record Date': record['date_submitted'].split()[0],
                        'Test Date': record['test_date'] or 'Manual',
                        'Input Method': record['input_method'],
                        **clinical_data,
                        **prediction_data
                    }
                    export_data.append(export_record)
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"diabetes_history_{st.session_state.user_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📧 Email Full History", use_container_width=True):
                st.info("📧 Email functionality would be implemented here!")
                st.write(f"Your complete history would be sent to: {st.session_state.user_email}")
        
    finally:
        conn.close()
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Main App"):
        st.session_state.current_page = 1
import database
from database import update_user_profile
def show_user_dashboard():
    """
    Displays the user dashboard for returning users to manage profile and view history.
    """
    st.title("👤 Your Dashboard")
    st.markdown("---")
    
    # Display current user info
    st.subheader("Your Profile Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**User ID:** `{st.session_state.user_id}`")
        st.info(f"**Current Name:** {st.session_state.user_name}")
    
    with col2:
        st.info(f"**Current Email:** {st.session_state.user_email}")
        st.info(f"**Member Since:** {st.session_state.get('date_created', 'N/A')}")
    
    # Profile editing section
    with st.expander("✏️ Edit Profile Information", expanded=False):
        st.warning("Update your name or email address below.")
        
        with st.form("profile_edit_form"):
            new_name = st.text_input("Full Name", value=st.session_state.user_name)
            new_email = st.text_input("Email Address", value=st.session_state.user_email)
            
            col_edit1, col_edit2 = st.columns(2)
            
            with col_edit1:
                update_submitted = st.form_submit_button("💾 Save Changes", use_container_width=True)
            
            with col_edit2:
                update_cancelled = st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary")
            
            if update_submitted:
                if new_name.strip() and new_email.strip():
                    conn = get_db_connection()
                    try:
                        update_user_profile(conn, st.session_state.user_id, new_name, new_email)
                        
                        # Update session state
                        st.session_state.user_name = new_name
                        st.session_state.user_email = new_email
                        
                        st.success("✅ Profile updated successfully!")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("❌ This email address is already registered to another user.")
                    except Exception as e:
                        st.error(f"❌ Error updating profile: {e}")
                    finally:
                        conn.close()
                else:
                    st.error("❌ Both fields are required.")
    
    st.markdown("---")
    
    # Action buttons
    st.subheader("What would you like to do?")
    
    col_action1, col_action2, col_action3 = st.columns(3)
    
    with col_action1:
        if st.button("📊 View My History", use_container_width=True, icon="📊"):
            st.session_state.current_page = 5  # Go to history page
            st.rerun()
    
    with col_action2:
        if st.button("🩺 New Prediction", use_container_width=True, icon="🩺", type="primary"):
            # Reset prediction state but keep user info
            keys_to_clear = ['prediction_made', 'diabetes_stage', 'diabetes_type', 'show_advanced', 
                            'stage_confidence', 'type_confidence', 'used_clinical_rules', 
                            'clinical_reason', 'full_report_data', 'extraction_status',
                            'record_saved', 'recommendations', 'current_page']
            for key in keys_to_clear:
                if key in st.session_state and key != 'current_page':
                    del st.session_state[key]
            # Clear input fields but keep user auth
            for field in ['age', 'gender', 'height', 'weight', 'bmi', 'bp_sys', 'bp_dia', 'chol',
                         'stress', 'hered', 'hba1c', 'fasting', 'post_meal', 'c_pep', 'ketones', 'antibodies']:
                if field in st.session_state:
                    st.session_state[field] = None
            st.session_state.user_raw_input = {}
            st.session_state.current_page = 1  # Start from basic info
            st.rerun()
    
    with col_action3:
        if st.button("🚪 Logout", use_container_width=True, icon="🚪", type="secondary"):
            # Clear all user-related session state
            for key in ['user_id', 'user_name', 'user_email', 'logged_in', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    # Quick stats preview
    conn = get_db_connection()
    try:
        records = get_user_health_records(conn, st.session_state.user_id, limit=5)
        
        if records:
            st.subheader("📈 Quick Stats Preview")
            
            # Count predictions
            prediction_counts = {'No Diabetes': 0, 'Pre-Diabetes': 0, 'Diabetes': 0}
            for record in records:
                prediction_data = json.loads(record['prediction_results'])
                stage = prediction_data.get('stage')
                if stage in prediction_counts:
                    prediction_counts[stage] += 1
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Records", len(records))
            
            with col_stat2:
                st.metric("Normal Results", prediction_counts['No Diabetes'])
            
            with col_stat3:
                st.metric("Risk Results", prediction_counts['Pre-Diabetes'] + prediction_counts['Diabetes'])
            
            st.caption(f"You have {len(records)} health record(s). Latest prediction was on {records[0]['date_submitted'].split()[0]}.")
    except Exception as e:
        st.error(f"Error loading quick stats: {e}")
    finally:
        conn.close()
    
    st.markdown("---")
    st.subheader("📋 Recent Lab Reports")

    conn = get_db_connection()
    try:
        lab_reports = conn.execute("""
        SELECT lrm.*, hr.date_submitted 
        FROM lab_reports_metadata lrm
        JOIN health_records hr ON lrm.record_id = hr.record_id
        WHERE hr.user_id = ? AND hr.input_method = 'lab_report'
        ORDER BY hr.date_submitted DESC
        LIMIT 5
        """, (st.session_state.user_id,)).fetchall()
        
        if lab_reports:
            for report in lab_reports:
                with st.expander(f"🔬 Lab Report from {report['date_submitted'].split()[0]}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if report['lab_name']:
                            st.write(f"**Lab:** {report['lab_name']}")
                        if report['pathologist_name']:
                            st.write(f"**Doctor:** {report['pathologist_name']}")
                    with col2:
                        if report['accession_number']:
                            st.write(f"**Report #:** {report['accession_number']}")
                        if report['test_date']:
                            st.write(f"**Test Date:** {report['test_date']}")
                    if st.button("View Full Report", key=f"view_report_{report['lab_id']}"):
                        st.session_state.viewing_lab_id = report['lab_id']
                        st.session_state.current_page = 5  # Go to history page
                        st.rerun()
        else:
            st.info("You don't have any lab reports yet. Upload a report to see it here!")
    except Exception as e:
        st.error(f"Error loading lab reports: {e}")
    finally:
        conn.close()

def generate_comprehensive_pdf(patient_name, user_input, prediction_data, feature_imp_df=None, feature_imp_fig=None, lab_data=None):
    """Generate a comprehensive PDF report with all details"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Set up styles
    title_style = ("Helvetica-Bold", 18)
    heading_style = ("Helvetica-Bold", 14)
    subheading_style = ("Helvetica-Bold", 12)
    normal_style = ("Helvetica", 10)
    small_style = ("Helvetica", 8)
    
    y_position = height - 50  # Start from top
    
    # Header
    c.setFont(*title_style)
    c.drawString(50, y_position, "Diabetes Risk Assessment Report")
    c.setFont(*small_style)
    c.drawString(width - 150, y_position, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_position -= 30
    c.line(50, y_position, width - 50, y_position)
    y_position -= 40
    
    # Patient Information Section
    c.setFont(*heading_style)
    c.drawString(50, y_position, "Patient Information")
    y_position -= 25
    
    c.setFont(*normal_style)
    info_lines = [
        f"Name: {patient_name}",
        f"User ID: {st.session_state.user_id}",
        f"Email: {st.session_state.user_email}",
        f"Age: {user_input.get('Age', 'N/A')} years",
        f"Gender: {user_input.get('Gender', 'N/A')}",
        f"Height: {user_input.get('Height_cm', 'N/A')} cm",
        f"Weight: {user_input.get('Weight_kg', 'N/A')} kg",
        f"BMI: {user_input.get('BMI', 'N/A')}"
    ]
    
    for line in info_lines:
        c.drawString(70, y_position, line)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = height - 50
    
    y_position -= 20
    
    # Health Metrics Section
    c.setFont(*heading_style)
    c.drawString(50, y_position, "Health Metrics")
    y_position -= 25
    
    c.setFont(*normal_style)
    health_metrics = [
        f"Blood Pressure: {user_input.get('BP_Systolic', 'N/A')}/{user_input.get('BP_Diastolic', 'N/A')} mmHg",
        f"Cholesterol: {user_input.get('Cholesterol', 'N/A')} mg/dL",
        f"Stress Level: {user_input.get('Stress', 'N/A')}",
        f"Family History: {'Yes' if user_input.get('Hereditary') == 1 else 'No' if user_input.get('Hereditary') == 0 else 'N/A'}"
    ]
    
    for metric in health_metrics:
        c.drawString(70, y_position, metric)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = height - 50
    
    y_position -= 20
    
    # Test Results Section
    c.setFont(*heading_style)
    c.drawString(50, y_position, "Test Results")
    y_position -= 25
    
    c.setFont(*normal_style)
    test_results = [
        f"HbA1c: {user_input.get('HbA1c', 'N/A')}%",
        f"Fasting Glucose: {user_input.get('Fasting_Glucose', 'N/A')} mg/dL",
        f"Post-Meal Glucose: {user_input.get('Post_Meal_Glucose', 'N/A')} mg/dL",
        f"C-Peptide: {user_input.get('C_Peptide', 'N/A')} ng/mL",
        f"Ketones: {user_input.get('Ketones', 'N/A')} mmol/L"
    ]
    
    for test in test_results:
        c.drawString(70, y_position, test)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = height - 50
    
    # Lab Information (if available)
    if lab_data and 'lab_info' in lab_data:
        y_position -= 20
        c.setFont(*heading_style)
        c.drawString(50, y_position, "Lab Information")
        y_position -= 25
        
        c.setFont(*normal_style)
        lab_info = lab_data['lab_info']
        lab_lines = []
        
        if 'Lab_Name' in lab_info:
            lab_lines.append(f"Lab Name: {lab_info['Lab_Name']}")
        if 'Lab_Address' in lab_info:
            lab_lines.append(f"Address: {lab_info['Lab_Address']}")
        if 'Test_Date' in lab_info:
            lab_lines.append(f"Test Date: {lab_info['Test_Date']}")
        if 'Lab_ID' in lab_info:
            lab_lines.append(f"Lab ID: {lab_info['Lab_ID']}")
        
        for line in lab_lines:
            c.drawString(70, y_position, line)
            y_position -= 20
            if y_position < 100:
                c.showPage()
                y_position = height - 50
    
    y_position -= 20
    
    # Prediction Results Section
    c.setFont(*heading_style)
    c.drawString(50, y_position, "Prediction Results")
    y_position -= 25
    
    c.setFont(*normal_style)
    prediction_text = f"Prediction: {prediction_data.get('stage', 'N/A')}"
    confidence_text = f"Confidence: {prediction_data.get('stage_confidence', 0) * 100:.1f}%"
    
    c.drawString(70, y_position, prediction_text)
    y_position -= 20
    c.drawString(70, y_position, confidence_text)
    y_position -= 20
    
    if prediction_data.get('type'):
        type_text = f"Diabetes Type: {prediction_data.get('type', 'N/A')}"
        type_confidence_text = f"Type Confidence: {prediction_data.get('type_confidence', 0) * 100:.1f}%"
        
        c.drawString(70, y_position, type_text)
        y_position -= 20
        c.drawString(70, y_position, type_confidence_text)
        y_position -= 20
    
    y_position -= 20
    
    # Feature Importance Section (if available)
    if feature_imp_df is not None and feature_imp_fig is not None:
        c.showPage()
        y_position = height - 50
        
        c.setFont(*heading_style)
        c.drawString(50, y_position, "Top Influencing Factors")
        y_position -= 30
        
        # Save figure to buffer and add to PDF
        img_buffer = io.BytesIO()
        feature_imp_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        # Add image to PDF
        img = ImageReader(img_buffer)
        img_width = 400
        img_height = 300
        c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
        y_position -= img_height + 20
        
        # Add feature importance table
        c.setFont(*subheading_style)
        c.drawString(50, y_position, "Feature Importance Scores:")
        y_position -= 20
        
        c.setFont(*small_style)
        for i, row in feature_imp_df.head(5).iterrows():
            c.drawString(70, y_position, f"{row['Feature']}: {row['Importance']:.4f}")
            y_position -= 15
            if y_position < 50:
                c.showPage()
                y_position = height - 50
    
    # Recommendations Section
    c.showPage()
    y_position = height - 50
    
    c.setFont(*heading_style)
    c.drawString(50, y_position, "Recommendations")
    y_position -= 30
    
    c.setFont(*normal_style)
    if 'recommendations' in st.session_state:
        recs = st.session_state.recommendations
        priority_colors = {'critical': '#ffcccc', 'important': '#fff3cd', 'supplementary': '#e2f0fd'}
        
        for priority in ['critical', 'important', 'supplementary']:
            if recs[priority]:
                c.setFillColor(priority_colors[priority])
                c.rect(50, y_position - 5, width - 100, 20, fill=True, stroke=False)
                c.setFillColorRGB(0, 0, 0)  # Reset to black text
                
                c.setFont(*subheading_style)
                priority_title = f"{priority.title()} Recommendations:"
                c.drawString(55, y_position, priority_title)
                y_position -= 25
                
                c.setFont(*normal_style)
                for rec in recs[priority]:
                    wrapped_lines = wrap(rec, width=80)
                    for line in wrapped_lines:
                        c.drawString(70, y_position, f"• {line}")
                        y_position -= 15
                        if y_position < 100:
                            c.showPage()
                            y_position = height - 50
                    y_position -= 5
                y_position -= 10
    
    # Footer
    c.showPage()
    y_position = height - 50
    c.setFont(*small_style)
    c.drawString(50, y_position, "Note: This report is AI-generated and should be reviewed by a healthcare professional.")
    c.drawString(50, y_position - 15, "Follow up with your doctor for proper diagnosis and treatment plan.")
    
    c.save()
    buffer.seek(0)
    return buffer

def send_email_with_pdf(sender_email, sender_password, receiver_email, patient_name, pdf_buffer):
    """Send the PDF report via email"""
    try:
        # Reset buffer position to beginning
        pdf_buffer.seek(0)
        
        # Read the PDF content once and store it
        pdf_content = pdf_buffer.read()
        
        msg = EmailMessage()
        msg['Subject'] = f"Diabetes Risk Assessment Report - {patient_name}"
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        msg.set_content(f"""
        Hi {patient_name},
        
        Please find attached your diabetes risk assessment report.
        
        This report includes:
        - Your health metrics and test results
        - AI prediction with confidence levels
        - Top factors influencing your risk
        - Personalized recommendations
        
        Please review this with your healthcare provider.
        
        Regards,
        Diabetes Predictor AI Team
        """)
        
        # Attach PDF using the stored content
        msg.add_attachment(
            pdf_content,
            maintype='application',
            subtype='pdf',
            filename=f"{patient_name.replace(' ', '_')}_Diabetes_Report.pdf"
        )
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Email failed: {str(e)}"
    
def send_email_with_csv(sender_email, sender_password, receiver_email, csv_data, filename):
    """Send CSV data via email as attachment"""
    try:
        msg = EmailMessage()
        msg['Subject'] = f"Your Complete Diabetes History - {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        msg.set_content(f"""
        Hi,
        
        Please find attached your complete diabetes history records.
        
        This file contains all your health assessments and predictions.
        
        Regards,
        Diabetes Predictor AI Team
        """)
        
        # Attach CSV
        msg.add_attachment(
            csv_data.encode('utf-8'),
            maintype='text',
            subtype='csv',
            filename=filename
        )
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        
        return True, "Email with history sent successfully!"
        
    except Exception as e:
        return False, f"Email failed: {str(e)}"
    

# Navigation functions
def go_to_page(page_number):
    st.session_state.current_page = page_number
    st.rerun()





if st.session_state.current_page == 0:
    # PHASE 1: Landing Page & Authentication
    show_landing_page()

# Page 1: Basic Information
elif st.session_state.current_page == 1:
    display_user_info()
    display_logout_button()
    st.title("🩺 Diabetes Risk Predictor")
    st.warning("⚠️ Important: This predictor is only accurate if you are not currently taking any diabetes medications.")
    st.markdown("""
        Enter your health details to check diabetes risk. 
        **Leave fields blank if you don't know the values** - the AI is trained to handle missing data!
        All predictions are AI-based and should be followed up with a doctor's opinion.
    """)
    
    st.markdown("---")
    
    st.subheader("📄 Lab Report Shortcut")
    st.info("Upload a lab report to automatically fill your details. You can still review and edit all fields manually.")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Upload your blood test report for automatic extraction",
        key="report_uploader"
    )
    
    if uploaded_file is not None and not st.session_state.extracted_values_processed:
            with st.spinner("🔍 Scanning your report... This might take a moment."):
                try:
                    raw_text = extract_values_from_report(uploaded_file)
                    extracted_data = extract_all_values(raw_text)
                    st.session_state.full_report_data = extracted_data
                    demographics = extracted_data.get('demographics', {})
                    tests = extracted_data.get('tests', {})
                    if 'Age' in demographics:
                        st.session_state.age = demographics['Age']
                    if 'Sex' in demographics:
                        st.session_state.gender = demographics['Sex']
                    test_mapping = {
    # --- HbA1c and related terms ---
    'HbA1c': 'hba1c',
    'Hb A1c': 'hba1c',
    'Hb A1 C': 'hba1c',
    'GLYCATED HAEMOGLOBIN ( HbA1c )': 'hba1c',
    'HBA1C': 'hba1c',
    'HBA1 C': 'hba1c',
    'A1c': 'hba1c',
    'A1 C': 'hba1c',
    'Glycated Hemoglobin': 'hba1c',
    'Glycosylated Hemoglobin': 'hba1c',
    'Glyco Hemoglobin': 'hba1c',
    'Glyco Hb': 'hba1c',
    'HgbA1c': 'hba1c',
    'Hgb A1c': 'hba1c',
    'Hgb A1 C': 'hba1c',
    'Non-diabetic': 'hba1c',
    'HbA1c (Glycosylated Hemoglobin)': 'hba1c',

    # --- Fasting glucose terms ---
    'Fasting Glucose': 'fasting',
    'Fasting Plasma Glucose': 'fasting',
    'FASTING PLASMA GLUCOSE': 'fasting',
    'Fasting Plasma Glucose Level': 'fasting',
    'Fasting Blood Glucose': 'fasting',
    'Fasting Blood Sugar': 'fasting',
    'Fasting Blood Sugar Level': 'fasting',
    'FBS': 'fasting',
    'FBG': 'fasting',
    'FPG': 'fasting',
    'Pre Breakfast Glucose': 'fasting',
    'Basal Blood Glucose': 'fasting',
    'FBS mg/dL': 'fasting',
    'Fasting (8–12 hrs)': 'fasting',
    'Glucose, Fasting': 'fasting',  # ADDED - This is the format in your report
    'Glucose Fasting': 'fasting',   # ADDED

    # --- Post-meal (Postprandial) glucose terms ---
    'Postprandial Glucose': 'post_meal',
    'Postprandial Blood Sugar': 'post_meal',
    'POST- PRANDIAL (PP) PLASMA GLUCOSE': 'post_meal',
    'PPBS': 'post_meal',
    'PPBG': 'post_meal',
    'PPG': 'post_meal',
    '2 hr PP': 'post_meal',
    '2 Hour PP': 'post_meal',
    '2 Hrs Post Meal': 'post_meal',
    'Post Lunch Blood Sugar': 'post_meal',
    'PLBS': 'post_meal',
    'PP2BS': 'post_meal',
    'Glucose Post Prandial': 'post_meal',
    'Post Meal Glucose': 'post_meal',
    'Post Meal Sugar': 'post_meal',
    'Post Meal (2 hr)': 'post_meal',
    'Post Lunch Glucose': 'post_meal',
    'After Meal Glucose': 'post_meal',
    'Pose Meal Glucose': 'post_meal',

    # --- Cholesterol terms ---
    'Cholesterol': 'chol',
    'Total Cholesterol': 'chol',
    'Serum Cholesterol': 'chol',
    'TC': 'chol',
    'Chol': 'chol',

    # --- C-Peptide terms ---
    'C-Peptide': 'c_pep',
    'C Peptide': 'c_pep',
    'Serum C-Peptide': 'c_pep',
    'Serum C Peptide': 'c_pep',
    'C-Peptide Level': 'c_pep',
    'C Peptide Level': 'c_pep',


    # --- Ketones terms ---
    'Ketones': 'ketones',
    'Urine Ketones': 'ketones',
    'Serum Ketones': 'ketones',
    'Blood Ketones': 'ketones',
    'Ketone Bodies': 'ketones',


    'ESTIMATED AVERAGE': 'eag',
    'eAG': 'eag',
    'Estimated Average Glucose': 'eag',
    'Average Estimated Glucose': 'eag',  # ADDED - This is in your report
                    }
                    if 'Non-diabetic' in tests and 'value' in tests['Non-diabetic']:
                        st.session_state.hba1c = tests['Non-diabetic']['value']

                    if 'POST-PRANDIAL (PP) PLASMA GLUCOSE' in tests:
                        st.session_state.post_meal = tests['POST-PRANDIAL (PP) PLASMA GLUCOSE']['value']

                    if 'FASTING PLASMA GLUCOSE' in tests:
                        st.session_state.fasting = tests['FASTING PLASMA GLUCOSE']['value']

                    if 'GLYCATED HAEMOGLOBIN ( HbA1c )' in tests:
                        st.session_state.hba1c = tests['GLYCATED HAEMOGLOBIN ( HbA1c )']['value']

# For eAG, either store it separately or ignore it since it's calculated from HbA1c
                    if 'ESTIMATED AVERAGE BLOOD GLUCOSE (eAG)' in tests:
    # Store it but don't use it for post_meal
                        st.session_state.eag = tests['ESTIMATED AVERAGE BLOOD GLUCOSE (eAG)']['value']
                
                    for test_name, test_data in tests.items():
                        clean_name = test_name.strip().lower()
                        value = test_data['value']
    
    # Add unit validation here
                        if any(term in clean_name for term in ['hba1c', 'a1c', 'glycated']):
        # HbA1c should be > 4% (not 0.04)
                            if value < 1:  # Likely decimal instead of percentage
                                value = value * 100  # Convert from decimal to percentage
    
                        elif any(term in clean_name for term in ['glucose', 'sugar', 'bs', 'ppbs', 'fbs']):
        # Glucose should be > 50 mg/dL (not 0.5)
                            if value < 10:  # Likely decimal instead of mg/dL
                                value = value * 100  # Convert from decimal
    
                        for pattern, field in test_mapping.items():
                            if pattern.lower() in clean_name:
                                setattr(st.session_state, field, value)
                                break
                    num_tests_found = len(tests)
                    num_demo_found = len(demographics)
                    if num_tests_found > 5:
                        st.session_state.extraction_status = 'success'
                        st.success(f"✅ Success! We found {num_tests_found} test values and {num_demo_found} patient details. The form has been updated. **Please review and confirm all values below.**")
                    elif num_tests_found > 0:
                        st.session_state.extraction_status = 'partial'
                        st.warning(f"⚠️ We found {num_tests_found} value(s). The form has been updated. Please review carefully and fill in any missing fields.")
                
                    else:
                        st.session_state.extraction_status = 'fail'
                        st.error("❌ We couldn't find any values in this report. Please enter your details manually or try a different report.")
                    st.session_state.extracted_values_processed = True
                    st.rerun()
                except Exception as e:
                    st.session_state.extraction_status = 'error'
                    st.error(f"❌ An error occurred during processing: {str(e)}. Please try manual entry.")
    if st.session_state.full_report_data and st.session_state.extraction_status in ['success', 'partial']:
        with st.expander("🔍 View Extracted Data (Debug)"):
            st.write("Demographics:", st.session_state.full_report_data.get('demographics', {}))
            st.write("Tests:", st.session_state.full_report_data.get('tests', {}))
            st.write("Lab Info:", st.session_state.full_report_data.get('lab_info', {}))
               
    st.markdown("---")
    st.subheader("👤 Basic Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, 
                        value=st.session_state.age if st.session_state.age is not None else None,
                        help="Leave blank if unknown", key='age_input')
        
        gender_options = ["Female", "Male"]
    # Get current gender from session state or default to first option
        current_gender = st.session_state.gender
        gender_index = gender_options.index(current_gender) if current_gender in gender_options else 0
        gender = st.radio("Gender", gender_options, index=gender_index, help="Select your gender", key='gender_input')
        
        height = st.number_input("Height (cm)", min_value=50.00, max_value=250.00, 
                           value= st.session_state.height if st.session_state.height is not None else None,
                           help="Leave blank if unknown", key='height_input')
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.00, max_value=200.00, 
                               value= st.session_state.weight,
                               help="Leave blank if unknown", key='weight_input')
        
        # Auto-calculate BMI if both height and weight are provided
        if height is not None and weight is not None and height > 0:
            bmi = weight / ((height/100) ** 2)
            st.metric("Calculated BMI", f"{bmi:.1f}")
            st.session_state.bmi = bmi
        else:
            if st.session_state.bmi is not None:
                st.metric("BMI", f"{st.session_state.bmi:.1f}")
            else:
                st.info("BMI will be calculated if height and weight are provided")
    
    # Navigation buttons
    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav2:
        if st.button("Next → Health Metrics", type="primary"):
            # Save inputs to session state
            st.session_state.age = age
            st.session_state.gender = gender
            st.session_state.height = height
            st.session_state.weight = weight
            save_inputs_to_session()
            go_to_page(2)

# Page 2: Health Metrics
elif st.session_state.current_page == 2:
    display_user_info()
    display_logout_button()
    st.title("❤️ Health Metrics")
    st.info("Provide values you know, leave blank if unknown")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        bp_systolic = st.number_input("BP Systolic", min_value=80.00, max_value=200.00, 
                                    value= st.session_state.bp_sys,
                                    help="Leave blank if unknown", key='bp_sys_input')
        bp_diastolic = st.number_input("BP Diastolic", min_value=50.00, max_value=120.00, 
                                     value= st.session_state.bp_dia,
                                     help="Leave blank if unknown", key='bp_dia_input')
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50.00, max_value=400.00, 
                                    value= st.session_state.chol,
                                    help="Leave blank if unknown", key='chol_input')
    
    with col4:
        stress_options = ["None", "Low", "Medium", "High"]
        stress_index = stress_options.index(st.session_state.stress) if st.session_state.stress in stress_options else 0
        stress = st.selectbox("Stress Level", stress_options, index=stress_index,
                            help="Select your stress level", key='stress_input')
        
        hereditary = st.radio("Family History of Diabetes", [0, 1], 
                            index=0 if st.session_state.hered == 0 else 1 if st.session_state.hered == 1 else 0,
                            format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "Unknown",
                            help="Select if you have family history of diabetes", key='hered_input')
        if hereditary is not None:
            st.info("💡 Family history increases diabetes risk")
    
    # Navigation buttons
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("← Back to Basic Info"):
            # Save inputs to session state
            st.session_state.bp_sys = bp_systolic
            st.session_state.bp_dia = bp_diastolic
            st.session_state.chol = cholesterol
            st.session_state.stress = stress
            st.session_state.hered = hereditary
            save_inputs_to_session()
            go_to_page(1)
    with col_nav3:
        if st.button("Next → Screening Tests", type="primary"):
            # Save inputs to session state
            st.session_state.bp_sys = bp_systolic
            st.session_state.bp_dia = bp_diastolic
            st.session_state.chol = cholesterol
            st.session_state.stress = stress
            st.session_state.hered = hereditary
            save_inputs_to_session()
            go_to_page(3)

# Page 3: Diabetes Screening Tests
elif st.session_state.current_page == 3:
    display_user_info()
    display_logout_button()
    st.title("🩸 Diabetes Screening Tests")
    st.info("Provide any test results you have available. Leave blank if unknown!")
    
    st.markdown("---")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        hba1c = st.number_input("HbA1c (%)", min_value=3.00, max_value=15.00, 
                          value= float(st.session_state.hba1c) if st.session_state.hba1c is not None else None,
                          step=0.1, help="Leave blank if unknown", key='hba1c_input')
        if hba1c is not None:
            status = "🔴 Diabetes" if hba1c > 6.4 else "🟡 Pre-Diabetes" if hba1c > 5.6 else "🟢 Normal"
            st.caption(f"{status} (<5.7%, 5.7-6.4%, >6.4%)")
    with col6:
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50.00, max_value=300.00, 
                                   value= float(st.session_state.fasting) if st.session_state.fasting is not None else None,
                                   help="Leave blank if unknown", key='fasting_input')
        if fasting_glucose is not None:
            status = "🔴 Diabetes" if fasting_glucose > 125 else "🟡 Pre-Diabetes" if fasting_glucose > 99 else "🟢 Normal"
            st.caption(f"{status} (<100, 100-125, >125)")
    
    with col7:
        post_meal_glucose = st.number_input("Post-Meal Glucose (mg/dL)", min_value=50.00, max_value=400.00, 
                                      value= float(st.session_state.post_meal) if st.session_state.post_meal is not None else None,
                                      help="Leave blank if unknown", key='post_meal_input')
        if post_meal_glucose is not None:
            status = "🔴 Diabetes" if post_meal_glucose > 199 else "🟡 Pre-Diabetes" if post_meal_glucose > 139 else "🟢 Normal"
            st.caption(f"{status} (<140, 140-199, >199)")
    
    # Navigation and Predict button
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("← Back to Health Metrics"):
            # Save inputs to session state
            st.session_state.hba1c = hba1c
            st.session_state.fasting = fasting_glucose
            st.session_state.post_meal = post_meal_glucose
            save_inputs_to_session()
            go_to_page(2)
    with col_nav3:
        if st.button("🔍 Predict Diabetes Risk", type="primary"):
            # Save inputs to session state
            st.session_state.hba1c = hba1c
            st.session_state.fasting = fasting_glucose
            st.session_state.post_meal = post_meal_glucose
            save_inputs_to_session()
            
            # Validate inputs
            is_valid, validation_message = validate_user_input(st.session_state.user_raw_input)
            if not is_valid:
                st.error(validation_message)
                st.info("💡 Tip: Provide at least some of these for a prediction: Height/Weight, Blood Pressure, Cholesterol, Family History, or any glucose test.")
            else:
                # Apply clinical rules first (for single test cases)
                clinical_prediction, clinical_confidence, clinical_reason = apply_clinical_rules(
                    st.session_state.user_raw_input.get('HbA1c'),
                    st.session_state.user_raw_input.get('Fasting_Glucose'),
                    st.session_state.user_raw_input.get('Post_Meal_Glucose')
                )
                
                try:
                    if clinical_prediction:
                        # Use clinical rule-based prediction
                        diabetes_stage = clinical_prediction
                        confidence = clinical_confidence
                        st.session_state.used_clinical_rules = True
                        st.session_state.clinical_reason = clinical_reason
                    else:
                        # Use ML model for complex cases
                        user_df = prepare_model_input(st.session_state.user_raw_input, feature_columns)
                        X_scaled = stage_scaler.transform(user_df)
                        stage_prediction = stage_model.predict(X_scaled)[0]
                        stage_proba = stage_model.predict_proba(X_scaled)[0]
                        diabetes_stage = stage_le.inverse_transform([stage_prediction])[0]
                        confidence = stage_proba[stage_prediction]
                        st.session_state.used_clinical_rules = False
                    
                    # Store in session state
                    st.session_state.prediction_made = True
                    st.session_state.diabetes_stage = diabetes_stage
                    st.session_state.stage_confidence = confidence
                    
                    # Generate and store recommendations
                    st.session_state.recommendations = generate_recommendations(st.session_state.user_raw_input, diabetes_stage, has_advanced_data=False)
                    
                    # Show advanced section if diabetes is predicted
                    if diabetes_stage == "Diabetes":
                        st.session_state.show_advanced = True
                    
                    # Go to results page
                    go_to_page(4)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("Please make sure you've provided at least some basic information.")

    

# Page 4: Results
elif st.session_state.current_page == 4:
    display_user_info()
    display_logout_button()
    st.title("📊 Your Results")
    
    if not st.session_state.prediction_made:
        st.warning("Please complete the prediction first.")
        if st.button("← Back to Screening Tests"):
            go_to_page(3)
        st.stop()
    if not st.session_state.get('record_saved', False):
        from database import create_health_record, create_lab_metadata
        conn = get_db_connection()
        try:
            # Determine input method and test date
            input_method = 'lab_report' if st.session_state.full_report_data else 'manual'
            test_date = None
            if st.session_state.full_report_data and 'lab_info' in st.session_state.full_report_data:
                test_date = st.session_state.full_report_data['lab_info'].get('Test_Date')
            
            # Prepare prediction results
            prediction_results = {
                'stage': st.session_state.diabetes_stage,
                'stage_confidence': st.session_state.stage_confidence,
                'type': st.session_state.diabetes_type,
                'type_confidence': st.session_state.type_confidence,
                'used_clinical_rules': st.session_state.used_clinical_rules,
                'clinical_reason': st.session_state.clinical_reason
            }
            
            
            record_id = create_health_record(
                conn, 
                st.session_state.user_id,
                input_method,
                test_date,
                st.session_state.user_raw_input,
                prediction_results
            )
            
           
            if st.session_state.full_report_data and 'lab_info' in st.session_state.full_report_data:
                create_lab_metadata(conn, record_id, st.session_state.full_report_data['lab_info'])
            
            st.session_state.record_saved = True
            st.success("✅ Your prediction has been saved to your history!")
            
        except Exception as e:
            st.error(f"❌ Error saving your record: {e}")
        finally:
            conn.close()
    
    diabetes_stage = st.session_state.diabetes_stage
    confidence = st.session_state.stage_confidence
    
    # Show if clinical rules were used
    if st.session_state.used_clinical_rules:
        st.info(f"🔬 {st.session_state.clinical_reason}")
    
    # Confidence indicator
    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
    st.markdown(f"**Confidence Level:** :{confidence_color}[{confidence:.1%}]")
    
    if diabetes_stage == "No Diabetes":
        st.success(f"✅ **Prediction: No Diabetes**")
        st.info("Your health parameters indicate normal glucose metabolism. Maintain your healthy habits!")
    elif diabetes_stage == "Pre-Diabetes":
        st.warning(f"⚠️ **Prediction: Pre-Diabetes**")
        st.info("Your results suggest impaired glucose metabolism. Lifestyle changes can help prevent progression to diabetes.")
    else:  # Diabetes
        st.error(f"🚨 **Prediction: Diabetes**")
        st.warning("Your health parameters indicate diabetes. Please consult with a healthcare professional for confirmation and treatment.")
    
    # Display recommendations
    display_recommendations(st.session_state.recommendations)
    
    # Show advanced section for diabetes cases
    if st.session_state.show_advanced:
        st.markdown("---")
        st.subheader("🔬 Advanced Diabetes Typing")
        
        with st.expander("Determine Diabetes Type (Optional)", expanded=True):
            st.info("Provide additional test results if available to determine diabetes type")
            
            col8, col9, col10 = st.columns(3)
            with col8:
                c_peptide = st.number_input("C-Peptide (ng/mL)", min_value=0.1, max_value=8.0, 
                                          value=st.session_state.c_pep, step=0.1,
                                          help="Leave blank if unknown", key='c_pep_input')
                if c_peptide is not None:
                    status = "🔴 Low (Type 1)" if c_peptide < 1.1 else "🟢 Normal" if c_peptide < 4.5 else "🟡 High (Type 2)"
                    st.caption(f"{status} (1.1-4.4 ng/mL)")
            
            with col9:
                ketones = st.number_input("Ketones (mmol/L)", min_value=0.0, max_value=8.0, 
                                        value=st.session_state.ketones, step=0.1,
                                        help="Leave blank if unknown", key='ketones_input')
                if ketones is not None:
                    status = "🟢 Normal" if ketones < 0.6 else "🟡 Elevated" if ketones < 3.0 else "🔴 High (Type 1)"
                    st.caption(f"{status} (<0.6 mmol/L)")
            
            with col10:
                antibodies = st.radio("Diabetes Antibodies", [0, 1], 
                                    index=0 if st.session_state.antibodies == 0 else 1 if st.session_state.antibodies == 1 else 0,
                                    format_func=lambda x: "Positive" if x == 1 else "Negative" if x == 0 else "Unknown",
                                    help="Select if known", key='antibodies_input')
                if antibodies is not None:
                    st.caption("Often positive in Type 1 diabetes")
            
            if st.button("🩺 Determine Diabetes Type", type="secondary"):
                # Update user input with advanced values
                st.session_state.c_pep = c_peptide
                st.session_state.ketones = ketones
                st.session_state.antibodies = antibodies
                save_inputs_to_session()
                
                # Prepare for type model
                try:
                    advanced_df = prepare_model_input(st.session_state.user_raw_input, feature_columns)
                    
                    # Scale and predict
                    X_type_scaled = type_scaler.transform(advanced_df)
                    type_prediction = type_model.predict(X_type_scaled)[0]
                    type_proba = type_model.predict_proba(X_type_scaled)[0]
                    
                    # Get confidence and decoded prediction
                    type_confidence = type_proba[type_prediction]
                    diabetes_type = type_le.inverse_transform([type_prediction])[0]
                    
                    # Store in session state
                    st.session_state.diabetes_type = diabetes_type
                    st.session_state.type_confidence = type_confidence

                    st.session_state.recommendations = generate_recommendations(
                        st.session_state.user_raw_input, 
                        diabetes_stage, 
                        has_advanced_data=True
                    )
                    
                    # Show type prediction
                    if diabetes_type == "Type 1":
                        st.error(f"🔬 **Type Prediction: Type 1 Diabetes** (Confidence: {type_confidence:.1%})")
                        st.info("This suggests autoimmune diabetes. Please consult an endocrinologist for proper management.")
                    else:
                        st.error(f"🔬 **Type Prediction: Type 2 Diabetes** (Confidence: {type_confidence:.1%})")
                        st.info("This suggests insulin resistance-related diabetes. Lifestyle changes and medication can help manage this condition.")
                        
                    st.markdown("---")
                    st.subheader("📋 Additional Tests for More Accurate Prediction")
                    display_recommendations(st.session_state.recommendations)
                except Exception as e:
                    st.error(f"❌ Error determining diabetes type: {str(e)}")
        
        # Show previous type prediction if available
        if st.session_state.diabetes_type:
            st.markdown("---")
            st.subheader("Previous Type Prediction")
            diabetes_type = st.session_state.diabetes_type
            type_confidence = st.session_state.type_confidence
            
            if diabetes_type == "Type 1":
                st.error(f"🔬 **Type Prediction: Type 1 Diabetes** (Confidence: {type_confidence:.1%})")
            else:
                st.error(f"🔬 **Type Prediction: Type 2 Diabetes** (Confidence: {type_confidence:.1%})")
    
    # Navigation button
    st.markdown("---")
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 1,1,1])
    with col_nav1:
        if st.button("← Back to Screening Tests", use_container_width=True):
            go_to_page(3)
    with col_nav2:
        if st.button("🔄 Start New Prediction", use_container_width=True, type="primary"):
        # Reset the extraction processed flag
            st.session_state.extracted_values_processed = False
        # Clear prediction-related session state but keep user info
            keys_to_clear = ['prediction_made', 'diabetes_stage', 'diabetes_type', 'show_advanced', 
                        'stage_confidence', 'type_confidence', 'used_clinical_rules', 
                        'clinical_reason', 'full_report_data', 'extraction_status',
                        'record_saved', 'recommendations']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        # Clear input fields
            for field in ['age', 'gender', 'height', 'weight', 'bmi', 'bp_sys', 'bp_dia', 'chol',
                     'stress', 'hered', 'hba1c', 'fasting', 'post_meal', 'c_pep', 'ketones', 'antibodies']:
                if field in st.session_state:
                    st.session_state[field] = None
            st.session_state.user_raw_input = {}
            go_to_page(1)
    with col_nav3:
        if st.button("📊 View My History", use_container_width=True, type="secondary"):
            st.session_state.current_page = 5
            st.rerun()
    with col_nav4:  # NEW BUTTON
        if st.button("📈 Detailed Analysis", use_container_width=True, type="secondary"):
            st.session_state.current_page = 7  # Go to new analysis page
            st.rerun()
    with col_nav5:  # NEW BUTTON
        if st.button("🌱 Lifestyle Tips", use_container_width=True, type="secondary"):
            st.session_state.current_page = 8
            st.rerun()
    if st.session_state.full_report_data and st.session_state.extraction_status in ['success', 'partial']:
        st.markdown("---")
        with st.expander("📄 Full Lab Report Analysis", expanded=True):
            st.info("The following values were identified in your report. This analysis is for informational purposes and is not part of the core AI prediction.")
        
            report_data = st.session_state.full_report_data
            demos = report_data.get('demographics', {})
            lab_info = report_data.get('lab_info', {})
            tests = report_data.get('tests', {})
        
        # Create a clean layout with columns
            col1, col2 = st.columns(2)
        
            with col1:
            # ===== LAB INFORMATION =====
                if lab_info:
                    st.subheader("🏥 Lab Information")
    
    # Display the Test Date first and make it prominent
                    if 'Test_Date' in lab_info:
                        st.markdown(f"**🕐 Test Date:** **{lab_info['Test_Date']}**")
                        st.markdown("---")
                    
                    if 'Lab_ID' in lab_info:
                        st.markdown(f"**🔢 Lab ID:** {lab_info['Lab_ID']}")

    # Display Main Lab Name if found
                    if 'Lab_Name' in lab_info:
                        st.markdown(f"**🏢 Main Laboratory:** {lab_info['Lab_Name']}")
                    
                    if 'Pathologist' in lab_info:
                        st.markdown(f"**👨‍⚕️ Pathologist:** Dr. {lab_info['Pathologist']}")
    
    # Display Collection Point (clinic/doctor) if found
                    if 'Collection_Point' in lab_info:
                        st.markdown(f"**📍 Collection Point:** {lab_info['Collection_Point']}")
    
    # Display Address
                    if 'Lab_Address' in lab_info:
                        st.markdown(f"**📫 Address:** {lab_info['Lab_Address']}")
                    elif 'Lab_Location' in lab_info:
                        st.markdown(f"**📫 Location:** {lab_info['Lab_Location']}")
    
                    st.markdown("---") # Add a separator after lab info
                else:
                    st.write("No lab information extracted")
            # ===== PATIENT DEMOGRAPHICS =====
                if demos:
                    st.subheader("👤 Patient Details")
                    demo_details = []
                
                    if 'Name' in demos:
                        demo_details.append(f"**Name:** {demos['Name']}")
                    if 'Age' in demos:
                        demo_details.append(f"**Age:** {demos['Age']} years")
                    if 'Sex' in demos:
                        demo_details.append(f"**Gender:** {demos['Sex']}")
                    if 'Patient_ID' in demos:
                        demo_details.append(f"**Patient ID:** {demos['Patient_ID']}")
                
                    for detail in demo_details:
                        st.write(detail)
                else:
                    st.write("No demographic information extracted")
        
            with col2:
            # ===== TESTS SUMMARY =====
                if tests:
                    st.subheader("🧪 Tests Summary")
                
                # Count tests by category
                    diabetes_tests = [name for name in tests.keys() if any(term in name.lower() for term in ['glucose', 'sugar', 'hba1c', 'a1c', 'glycated'])]
                    lipid_tests = [name for name in tests.keys() if any(term in name.lower() for term in ['cholesterol', 'hdl', 'ldl', 'triglyceride', 'lipid'])]
                    other_tests = [name for name in tests.keys() if name not in diabetes_tests + lipid_tests]
                
                    st.metric("Total Tests Found", len(tests))
                    st.metric("Diabetes-related Tests", len(diabetes_tests))
                    st.metric("Other Tests", len(other_tests))
            
                st.markdown("---")
            
            # ===== EXTRACTION STATUS =====
                st.subheader("📊 Extraction Quality")
                if st.session_state.extraction_status == 'success':
                    st.success("✅ High quality extraction")
                    st.write("Most values were successfully extracted")
                elif st.session_state.extraction_status == 'partial':
                    st.warning("⚠️ Partial extraction")
                    st.write("Some values were extracted, but some may be missing")
                else:
                    st.info("ℹ️ Basic extraction")
                    st.write("Limited information was extracted")
        
            st.markdown("---")
        
        # ===== DETAILED TESTS TABLE =====
            if tests:
                st.subheader("🔬 Detailed Test Results")
            
            # Define known ranges for common tests with better organization
                known_ranges = {
                # Diabetes tests
                    'HbA1c': (4.0, 5.6, 6.4, '%', "Diabetes monitoring"),
                    'Hb A1c': (4.0, 5.6, 6.4, '%', "Diabetes monitoring"),
                    'GLYCATED HAEMOGLOBIN': (4.0, 5.6, 6.4, '%', "Diabetes monitoring"),
                    'Fasting Glucose': (70, 99, 125, 'mg/dL', "Diabetes screening"),
                    'Fasting Plasma Glucose': (70, 99, 125, 'mg/dL', "Diabetes screening"),
                    'Postprandial Glucose': (70, 140, 199, 'mg/dL', "Diabetes screening"),
                    'PPBS': (70, 140, 199, 'mg/dL', "Post-meal glucose"),
                    'Estimated Average Glucose': (90, 117, 140, 'mg/dL', "Calculated from HbA1c"),
                
                # Lipid tests
                    'Cholesterol': (0, 200, 240, 'mg/dL', "Heart health"),
                    'Total Cholesterol': (0, 200, 240, 'mg/dL', "Heart health"),
                    'HDL': (40, 60, None, 'mg/dL', "Good cholesterol (higher better)"),
                    'LDL': (None, 100, 160, 'mg/dL', "Bad cholesterol (lower better)"),
                    'Triglycerides': (0, 150, 200, 'mg/dL', "Blood fats"),
                
                # Other common tests
                    'Vitamin D': (30, None, 100, 'ng/mL', "Bone health"),
                    'C Peptide': (1.1, 4.4, None, 'ng/mL', "Diabetes type differentiation"),
                    'Creatinine': (0.6, 1.1, None, 'mg/dL', "Kidney function"),
            }
            
            # Create a sorted list of tests (diabetes-related first, then others)
                sorted_tests = []
                for test_name in tests:
                    test_lower = test_name.lower()
                    if any(term in test_lower for term in ['glucose', 'sugar', 'hba1c', 'a1c', 'glycated']):
                        sorted_tests.insert(0, test_name)  
                    else:
                        sorted_tests.append(test_name)
            
            # Display tests in a clean format
                for test_name in sorted_tests:
                    data = tests[test_name]
                    value = data['value']
                    unit = data.get('unit', '')
                
                # Create a container for each test
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 1, 2])
                    
                        with col_a:
                            st.write(f"**{test_name}**")
                    
                        with col_b:
                            st.write(f"**{value} {unit}**")
                    
                        with col_c:
                            if test_name in known_ranges:
                                low, normal, high, expected_unit, description = known_ranges[test_name]
                            
                            # Determine status
                                if high is not None and value >= high:
                                    status = "🔴 High"
                                    color = "#ffcccc"
                                elif low is not None and value <= low:
                                    status = "🔴 Low" 
                                    color = "#ffcccc"
                                else:
                                    status = "✅ Normal"
                                    color = "#d4edda"
                            
                            # Display status and info
                                st.markdown(f"""
                                <div style='background-color: {color}; padding: 5px; border-radius: 5px;'>
                                {status} | {description}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show range if available
                                range_text = ""
                                if low is not None and high is not None:
                                    range_text = f"Normal: {low}-{normal} {expected_unit}"
                                elif low is not None:
                                    range_text = f"Normal: >{low} {expected_unit}"
                                elif high is not None:
                                    range_text = f"Normal: <{high} {expected_unit}"
                            
                                if range_text:
                                     st.caption(range_text)
                            else:
                            # For unknown tests
                                st.info("ℹ️ No reference range available")
                                st.caption("Consult your doctor for interpretation")
                    
                        st.markdown("---")
        
            else:
                st.warning("No test results were extracted from the report")
        
        # ===== RAW DATA FOR DEBUGGING (collapsed by default) =====
            with st.expander("🔍 View Raw Extracted Data (For Debugging)", expanded=False):
                st.write("**Demographics:**", demos)
                st.write("**Lab Info:**", lab_info)
                st.write("**Tests:**", tests)

        if st.button("🔄 Start Over", type="primary"):
    # Reset the extraction processed flag
            st.session_state.extracted_values_processed = False
            for key in list(st.session_state.keys()):
                if key not in ['patient_id', 'lab_values']:
                    del st.session_state[key]
            go_to_page(1)
    st.markdown("---")
    st.subheader("📄 Report Options")
    with st.expander("👤 Your Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.user_name}")
            st.write(f"**Age:** {st.session_state.user_raw_input.get('Age', 'Not provided')}")
            st.write(f"**Gender:** {st.session_state.user_raw_input.get('Gender', 'Not provided')}")
        with col2:
            st.write(f"**Height:** {st.session_state.user_raw_input.get('Height_cm', 'Not provided')} cm")
            st.write(f"**Weight:** {st.session_state.user_raw_input.get('Weight_kg', 'Not provided')} kg")
            st.write(f"**BMI:** {st.session_state.user_raw_input.get('BMI', 'Not provided')}")
    with st.expander("🧪 Test Results", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**HbA1c:** {st.session_state.user_raw_input.get('HbA1c', 'Not provided')}%")
            st.write(f"**Fasting Glucose:** {st.session_state.user_raw_input.get('Fasting_Glucose', 'Not provided')} mg/dL")
            st.write(f"**Post-Meal Glucose:** {st.session_state.user_raw_input.get('Post_Meal_Glucose', 'Not provided')} mg/dL")
        with col2:
            st.write(f"**C-Peptide:** {st.session_state.user_raw_input.get('C_Peptide', 'Not provided')} ng/mL")
            st.write(f"**Ketones:** {st.session_state.user_raw_input.get('Ketones', 'Not provided')} mmol/L")
            st.write(f"**Antibodies:** {'Positive' if st.session_state.user_raw_input.get('Antibodies') == 1 else 'Negative' if st.session_state.user_raw_input.get('Antibodies') == 0 else 'Not provided'}")
    if st.session_state.full_report_data and 'lab_info' in st.session_state.full_report_data:
        with st.expander("🏥 Lab Information", expanded=False):
            lab_info = st.session_state.full_report_data['lab_info']
            if 'Lab_Name' in lab_info:
                st.write(f"**Lab Name:** {lab_info['Lab_Name']}")
            if 'Lab_Address' in lab_info:
                st.write(f"**Address:** {lab_info['Lab_Address']}")
            if 'Test_Date' in lab_info:
                st.write(f"**Test Date:** {lab_info['Test_Date']}")
            if 'Pathologist' in lab_info:
                st.write(f"**Pathologist:** {lab_info['Pathologist']}")
    st.markdown("---")
    st.subheader("📧 Export Report")

# Test credentials first
    def test_email_connection():
        with st.expander("🔧 Test Email Settings First", expanded=False):
            with st.form("test_email_form"):
                test_sender = st.text_input("Test Email (Gmail)")
                test_password = st.text_input("Test App Password", type="password")
            
                if st.form_submit_button("Test Connection"):
                    if test_sender and test_password:
                        try:
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                                smtp.login(test_sender, test_password)
                            st.success("✅ Connection successful! Credentials work.")
                            return True
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
                            return False
                    else:
                        st.warning("Please enter both email and password")
        return None

    test_email_connection()

    col_pdf1, col_pdf2 = st.columns(2)

    with col_pdf1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                try:
                    prediction_data = {
                    'stage': st.session_state.diabetes_stage,
                    'stage_confidence': st.session_state.stage_confidence,
                    'type': st.session_state.diabetes_type,
                    'type_confidence': st.session_state.type_confidence
                    }
                
                    pdf_buffer = generate_comprehensive_pdf(
                    st.session_state.user_name,
                    st.session_state.user_raw_input,
                    prediction_data,
                    st.session_state.get('feature_imp_df'),
                    st.session_state.get('feature_imp_fig'),
                    st.session_state.full_report_data
                    )
                
                    st.session_state.pdf_buffer = pdf_buffer
                    st.success("✅ PDF report generated successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")

    with col_pdf2:
        if st.button("📤 Email PDF Report", use_container_width=True):
            if 'pdf_buffer' not in st.session_state:
                st.warning("Please generate the PDF first.")
            else:
                with st.form("email_form"):
                    st.info("Enter email details to send the report")
                    sender_email = st.text_input("Your Gmail")
                    sender_password = st.text_input("App Password", type="password", 
                                          help="16-character Gmail App Password (not regular password)")
                    receiver_email = st.text_input("Recipient Email", st.session_state.user_email)
                
                    if st.form_submit_button("Send Email", use_container_width=True):
                        if sender_email and sender_password and receiver_email:
                            try:
            # Use the function instead of direct implementation
                                success, message = send_email_with_pdf(
                                sender_email, sender_password, receiver_email, 
                                st.session_state.user_name, st.session_state.pdf_buffer
                                )
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"❌ Email failed: {e}")
                        else:
                            st.warning("Please fill all email fields.")

# Download button
    if 'pdf_buffer' in st.session_state:
        st.session_state.pdf_buffer.seek(0)
        st.download_button(
        "⬇️ Download PDF Report",
        st.session_state.pdf_buffer,
        file_name=f"{st.session_state.user_name}_Diabetes_Report.pdf",
        mime="application/pdf",
        use_container_width=True
        )

elif st.session_state.current_page == 5:
        display_user_info()
        display_logout_button()  # NEW PAGE!
        show_history_dashboard()
elif st.session_state.current_page == 6:
        display_user_info()  # NEW DASHBOARD PAGE!
        show_user_dashboard()

# ==================================================================
# PAGE 7: Detailed Analysis
# ==================================================================
elif st.session_state.current_page == 7:
    display_user_info()
    display_logout_button()
    st.title("📈 Detailed Analysis")
    st.markdown("---")
    
    if not st.session_state.prediction_made:
        st.warning("No prediction data available. Please complete a prediction first.")
        if st.button("← Back to Main App"):
            st.session_state.current_page = 1
        st.stop()
    
    # Display Class Probabilities
    st.header("🎯 Prediction Confidence Breakdown")
    
    # For Stage Model
    if st.session_state.diabetes_stage:
        st.subheader("Diabetes Stage Probabilities")
        # Create dummy probabilities for demonstration - YOU WILL NEED TO REPLACE THIS
        # In your actual code, you should get these from your model's predict_proba() method
        stage_classes = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
        stage_probs = [0.25, 0.35, 0.40]  # Replace with actual probabilities from your model
        
        stage_prob_df = pd.DataFrame({
            "Class": stage_classes,
            "Probability (%)": [p * 100 for p in stage_probs]
        })
        st.dataframe(stage_prob_df.style.format({"Probability (%)": "{:.2f}%"}).highlight_max(axis=0, color="#ffcccc"), 
                    use_container_width=True, hide_index=True)
    
    # For Type Model (if diabetes was predicted)
    if st.session_state.diabetes_type and st.session_state.diabetes_stage == "Diabetes":
        st.subheader("Diabetes Type Probabilities")
        # Create dummy probabilities for demonstration - YOU WILL NEED TO REPLACE THIS
        type_classes = ["Type 1", "Type 2"]
        type_probs = [0.30, 0.70]  # Replace with actual probabilities from your model
        
        type_prob_df = pd.DataFrame({
            "Class": type_classes,
            "Probability (%)": [p * 100 for p in type_probs]
        })
        st.dataframe(type_prob_df.style.format({"Probability (%)": "{:.2f}%"}).highlight_max(axis=0, color="#ffcccc"), 
                    use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Display Feature Importance
    st.header("🧠 Top Influencing Factors")
    st.info("These are the health parameters that most influenced your diabetes risk prediction.")
    
    # Create sample feature importance data - YOU WILL NEED TO REPLACE THIS
    # In your actual code, you should get feature importance from your model
    features = ['HbA1c', 'Fasting_Glucose', 'BMI', 'Age', 'Post_Meal_Glucose', 
                'Hereditary', 'Cholesterol', 'BP_Systolic', 'C_Peptide', 'Stress']
    importance_scores = [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    feature_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feature_imp_df))
    ax.barh(y_pos, feature_imp_df['Importance'], color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_imp_df['Feature'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Most Influential Features')
    
    # Add value labels on bars
    for i, v in enumerate(feature_imp_df['Importance']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    st.pyplot(fig)
    
    # Display as table
    st.markdown("**Feature Importance Scores:**")
    st.dataframe(feature_imp_df.sort_values('Importance', ascending=False).style.format({"Importance": "{:.4f}"}), 
                use_container_width=True, hide_index=True)
    
    # Store for PDF generation
    st.session_state.feature_imp_df = feature_imp_df
    st.session_state.feature_imp_fig = fig
    
    st.markdown("---") 
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            go_to_page(4)
    with col2:
        if st.button("🔄 New Prediction", use_container_width=True, type="primary"):
            # Reset and go to start
            keys_to_clear = ['prediction_made', 'diabetes_stage', 'diabetes_type', 'show_advanced', 
                           'stage_confidence', 'type_confidence', 'used_clinical_rules', 
                           'clinical_reason', 'full_report_data', 'extraction_status',
                           'record_saved', 'recommendations']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            for field in ['age', 'gender', 'height', 'weight', 'bmi', 'bp_sys', 'bp_dia', 'chol',
                         'stress', 'hered', 'hba1c', 'fasting', 'post_meal', 'c_pep', 'ketones', 'antibodies']:
                if field in st.session_state:
                    st.session_state[field] = None
            st.session_state.user_raw_input = {}
            go_to_page(1)
else:
    st.warning("Please log in first.")
    st.session_state.current_page = 0
    st.rerun()


# Footer
st.markdown("---")
st.caption("💡 Tip: The more information you provide, the more accurate your prediction will be. But don't worry if you're missing some values - our AI is trained to handle incomplete data!")