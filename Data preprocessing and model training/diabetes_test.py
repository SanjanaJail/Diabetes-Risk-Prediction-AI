import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load your new models and components
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

# Initialize session state for the two-stage approach
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

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🩺 Diabetes Risk Predictor")
st.warning("⚠️ Important: This predictor is only accurate if you are not currently taking any diabetes medications.")
st.markdown("""
    Enter your health details to check diabetes risk. 
    **Leave fields blank if you don't know the values** - the AI is trained to handle missing data!
     All predictions are AI-based and should be followed up with a doctor's opinion.
""")

with st.sidebar:
    # Mini quick instructions (always visible)
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
    
    # Full instructions (collapsible)
    with st.expander("📖 Detailed Instructions", expanded=True):
        st.markdown("""
        - Fill in **basic information** you know (Age, Gender, Height, Weight)  
        - Provide **health metrics** you know: BP, Cholesterol, Stress Level, Family History  
        - Enter **any screening test results** you have - leave others blank!  
        - Click **'Predict Diabetes Risk'** to get your result  
        - If Diabetes is detected, provide **additional tests** if available
        """)
    
    st.caption("⚠️ This is an AI-based risk assessment tool and should not replace medical consultation.")

# Progress Indicator
st.markdown("### 📋 Progress")
col1, col2, col3 = st.columns(3)
with col1:
    st.success("Step 1: Basic Information")
with col2:
    status = "✅" if st.session_state.prediction_made else "➡️"
    st.info(f"Step 2: Results {status}")
with col3:
    advanced_status = "➡️" if st.session_state.show_advanced else "⏭️"
    st.write(f"Step 3: Advanced Tests {advanced_status}")

st.markdown("---")

def apply_clinical_rules(hba1c, fasting_glucose, post_meal_glucose):
    """Apply WHO/ADA clinical guidelines when only limited tests are available"""
    # Count how many glucose tests are provided
    provided_tests = sum([x is not None for x in [hba1c, fasting_glucose, post_meal_glucose]])
    
    # If only one test is provided, apply clinical rules
    if provided_tests == 1:
        if hba1c is not None:
            if hba1c >= 6.5:
                return "Diabetes", 0.95, "Clinical Rule: HbA1c ≥ 6.5% = Diabetes"
            elif hba1c >= 5.7:
                return "Pre-Diabetes", 0.90, "Clinical Rule: HbA1c 5.7-6.4% = Pre-Diabetes"
            else:
                return "No Diabetes", 0.90, "Clinical Rule: HbA1c < 5.7% = Normal"
                
        elif fasting_glucose is not None:
            if fasting_glucose >= 126:
                return "Diabetes", 0.95, "Clinical Rule: Fasting Glucose ≥ 126 mg/dL = Diabetes"
            elif fasting_glucose >= 100:
                return "Pre-Diabetes", 0.90, "Clinical Rule: Fasting Glucose 100-125 mg/dL = Pre-Diabetes"
            else:
                return "No Diabetes", 0.90, "Clinical Rule: Fasting Glucose < 100 mg/dL = Normal"
                
        elif post_meal_glucose is not None:
            if post_meal_glucose >= 200:
                return "Diabetes", 0.95, "Clinical Rule: Post-Meal Glucose ≥ 200 mg/dL = Diabetes"
            elif post_meal_glucose >= 140:
                return "Pre-Diabetes", 0.90, "Clinical Rule: Post-Meal Glucose 140-199 mg/dL = Pre-Diabetes"
            else:
                return "No Diabetes", 0.90, "Clinical Rule: Post-Meal Glucose < 140 mg/dL = Normal"
    
    # If multiple tests or no tests, let ML model decide
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
        if user_raw_input['HbA1c'] is None:
            recommendations['critical'].append("HbA1c test - crucial for diabetes diagnosis")
        if user_raw_input['Fasting_Glucose'] is None:
            recommendations['critical'].append("Fasting Glucose test - essential for diabetes screening")
        if user_raw_input['Post_Meal_Glucose'] is None:
            recommendations['critical'].append("Post-Meal Glucose test - important for comprehensive assessment")
        
        # IMPORTANT: Key risk factors and type differentiation
        if user_raw_input['Age'] is None:
            recommendations['important'].append("Age - helps assess age-related risk factors")
        if user_raw_input['BMI'] is None and (user_raw_input['Height_cm'] is None or user_raw_input['Weight_kg'] is None):
            recommendations['important'].append("BMI calculation - important weight-related risk assessment")
        if user_raw_input['Hereditary'] is None:
            recommendations['important'].append("Family history - helps assess genetic predisposition")
        if user_raw_input['BP_Systolic'] is None or user_raw_input['BP_Diastolic'] is None:
            recommendations['important'].append("Blood Pressure - hypertension is a diabetes risk factor")
    
    # For diabetes cases: advanced tests for type differentiation
    if diabetes_stage == "Diabetes":
        # If user has started advanced testing OR we're in post-advanced context
        if has_advanced_data or any([
            user_raw_input['C_Peptide'] is not None,
            user_raw_input['Ketones'] is not None, 
            user_raw_input['Antibodies'] is not None
        ]):
            # Check which advanced tests are still missing
            if user_raw_input['C_Peptide'] is None:
                recommendations['critical'].append("C-Peptide test - crucial for determining diabetes type (Type 1 vs Type 2)")
            if user_raw_input['Ketones'] is None:
                recommendations['important'].append("Ketones test - helps identify Type 1 diabetes risk")
            if user_raw_input['Antibodies'] is None:
                recommendations['important'].append("Diabetes Antibodies test - important for Type 1 diabetes detection")
        else:
            # If no advanced data provided yet, only suggest basic diabetes confirmation tests
            recommendations['important'].append("Advanced diabetes tests (C-Peptide, Ketones, Antibodies) - for type differentiation if needed")
    
    # Only show supplementary recommendations if user hasn't started advanced testing
    if not has_advanced_data:
        # SUPPLEMENTARY: Additional context
        if user_raw_input['Cholesterol'] is None:
            recommendations['supplementary'].append("Cholesterol levels - provides complete cardiovascular health picture")
        if user_raw_input['Stress'] is None:
            recommendations['supplementary'].append("Stress level - stress can affect glucose metabolism")
        if user_raw_input['Gender'] is None:
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
    # These are the fields that alone don't provide meaningful diabetes prediction power
    non_informative_fields = ['Age', 'Gender', 'Stress']
    informative_fields_provided = any(
        user_raw_input[field] is not None 
        for field in user_raw_input 
        if field not in non_informative_fields
    )
    
    if not informative_fields_provided:
        return False, "⚠️ Please provide at least one health metric (Height/Weight, Blood Pressure, Cholesterol, Family History, or any glucose test) for a meaningful prediction."
    
    # Rule 3: Check if we have at least SOME health data
    health_fields = ['Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 
                    'Cholesterol', 'Hereditary', 'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose']
    health_data_provided = any(user_raw_input[field] is not None for field in health_fields)
    
    if not health_data_provided:
        return False, "⚠️ Please provide some health information like Height/Weight, Blood Pressure, or any test results for diabetes assessment."
    
    # All validation passed
    return True, "✅ Sufficient information provided for prediction."


# Main input form
with st.form("diabetes_prediction_form"):
    st.subheader("👤 Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=None, 
                            help="Leave blank if unknown", key='age')
        gender = st.radio("Gender", ["Female", "Male"], index=None, 
                         help="Select your gender", key='gender')
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=None,
                               help="Leave blank if unknown", key='height')
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=None,
                               help="Leave blank if unknown", key='weight')
        # Auto-calculate BMI if both height and weight are provided
        if height is not None and weight is not None and height > 0:
            bmi = weight / ((height/100) ** 2)
            st.metric("Calculated BMI", f"{bmi:.1f}")
        else:
            bmi = None
            st.info("BMI will be calculated if height and weight are provided")

    st.markdown("---")
    st.subheader("❤️ Health Metrics")
    col3, col4 = st.columns(2)
    
    with col3:
        bp_systolic = st.number_input("BP Systolic", min_value=80, max_value=200, value=None,
                                    help="Leave blank if unknown", key='bp_sys')
        bp_diastolic = st.number_input("BP Diastolic", min_value=50, max_value=120, value=None,
                                     help="Leave blank if unknown", key='bp_dia')
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=400, value=None,
                                    help="Leave blank if unknown", key='chol')
    
    with col4:
        stress = st.selectbox("Stress Level", ["None", "Low", "Medium", "High"], index=None,
                            help="Select your stress level", key='stress')
        hereditary = st.radio("Family History of Diabetes", [0, 1], index=None,
                            format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "Unknown",
                            help="Select if you have family history of diabetes", key='hered')
        if hereditary is not None:
            st.info("💡 Family history increases diabetes risk")

    st.markdown("---")
    st.subheader("🩸 Diabetes Screening Tests")
    st.info("Provide any test results you have available. Leave blank if unknown!")
    
    col5, col6, col7 = st.columns(3)
    with col5:
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=None, step=0.1,
                              help="Leave blank if unknown", key='hba1c')
        if hba1c is not None:
            status = "🔴 Diabetes" if hba1c > 6.4 else "🟡 Pre-Diabetes" if hba1c > 5.6 else "🟢 Normal"
            st.caption(f"{status} (<5.7%, 5.7-6.4%, >6.4%)")
    
    with col6:
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=300, value=None,
                                       help="Leave blank if unknown", key='fasting')
        if fasting_glucose is not None:
            status = "🔴 Diabetes" if fasting_glucose > 125 else "🟡 Pre-Diabetes" if fasting_glucose > 99 else "🟢 Normal"
            st.caption(f"{status} (<100, 100-125, >125)")
    
    with col7:
        post_meal_glucose = st.number_input("Post-Meal Glucose (mg/dL)", min_value=50, max_value=400, value=None,
                                          help="Leave blank if unknown", key='post_meal')
        if post_meal_glucose is not None:
            status = "🔴 Diabetes" if post_meal_glucose > 199 else "🟡 Pre-Diabetes" if post_meal_glucose > 139 else "🟢 Normal"
            st.caption(f"{status} (<140, 140-199, >199)")

    # Submit button for stage 1
    predict_button = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary")

# Handle stage 1 prediction
if predict_button:
    # Collect all user inputs
    user_raw_input = {
        'Age': age,
        'Gender': gender,
        'Height_cm': height,
        'Weight_kg': weight,
        'BMI': bmi,
        'BP_Systolic': bp_systolic,
        'BP_Diastolic': bp_diastolic,
        'Cholesterol': cholesterol,
        'Stress': stress,
        'Hereditary': hereditary,
        'HbA1c': hba1c,
        'Fasting_Glucose': fasting_glucose,
        'Post_Meal_Glucose': post_meal_glucose,
        'C_Peptide': None,  # Will be filled in advanced section if needed
        'Ketones': None,    # Will be filled in advanced section if needed
        'Antibodies': None  # Will be filled in advanced section if needed
    }
    is_valid, validation_message = validate_user_input(user_raw_input)
    if not is_valid:
        st.error(validation_message)
        st.info("💡 Tip: Provide at least some of these for a prediction: Height/Weight, Blood Pressure, Cholesterol, Family History, or any glucose test.")
        st.stop()  
    
    # Apply clinical rules first (for single test cases)
    clinical_prediction, clinical_confidence, clinical_reason = apply_clinical_rules(
        hba1c, fasting_glucose, post_meal_glucose
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
            user_df = prepare_model_input(user_raw_input, feature_columns)
            X_scaled = stage_scaler.transform(user_df)
            stage_prediction = stage_model.predict(X_scaled)[0]
            stage_proba = stage_model.predict_proba(X_scaled)[0]
            diabetes_stage = stage_le.inverse_transform([stage_prediction])[0]
            confidence = stage_proba[stage_prediction]
            st.session_state.used_clinical_rules = False
        
        # Store in session state
        st.session_state.user_raw_input = user_raw_input
        st.session_state.prediction_made = True
        st.session_state.diabetes_stage = diabetes_stage
        st.session_state.stage_confidence = confidence
        
        # Generate and store recommendations
        st.session_state.recommendations = generate_recommendations(user_raw_input, diabetes_stage, has_advanced_data=False)
        
        # Show advanced section if diabetes is predicted
        if diabetes_stage == "Diabetes":
            st.session_state.show_advanced = True
            
        st.success("✅ Prediction completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please make sure you've provided at least some basic information.")

# Show results after prediction
if st.session_state.prediction_made:
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
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
                c_peptide = st.number_input("C-Peptide (ng/mL)", min_value=0.1, max_value=8.0, value=None, step=0.1,
                                          help="Leave blank if unknown", key='c_pep')
                if c_peptide is not None:
                    status = "🔴 Low (Type 1)" if c_peptide < 1.1 else "🟢 Normal" if c_peptide < 4.5 else "🟡 High (Type 2)"
                    st.caption(f"{status} (1.1-4.4 ng/mL)")
            
            with col9:
                ketones = st.number_input("Ketones (mmol/L)", min_value=0.0, max_value=8.0, value=None, step=0.1,
                                        help="Leave blank if unknown", key='ketones')
                if ketones is not None:
                    status = "🟢 Normal" if ketones < 0.6 else "🟡 Elevated" if ketones < 3.0 else "🔴 High (Type 1)"
                    st.caption(f"{status} (<0.6 mmol/L)")
            
            with col10:
                antibodies = st.radio("Diabetes Antibodies", [0, 1], index=None,
                                    format_func=lambda x: "Positive" if x == 1 else "Negative" if x == 0 else "Unknown",
                                    help="Select if known", key='antibodies')
                if antibodies is not None:
                    st.caption("Often positive in Type 1 diabetes")
            
            if st.button("🩺 Determine Diabetes Type", type="secondary"):
                # Update user input with advanced values
                advanced_input = st.session_state.user_raw_input.copy()
                advanced_input['C_Peptide'] = c_peptide
                advanced_input['Ketones'] = ketones
                advanced_input['Antibodies'] = antibodies
                
                # Prepare for type model
                try:
                    advanced_df = prepare_model_input(advanced_input, feature_columns)
                    
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
                    st.session_state.user_raw_input = advanced_input

                    st.session_state.recommendations = generate_recommendations(
                        advanced_input, 
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

# Reset button
if st.session_state.prediction_made:
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            if key not in ['patient_id', 'lab_values']:  # Keep some states if needed
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.caption("💡 Tip: The more information you provide, the more accurate your prediction will be. But don't worry if you're missing some values - our AI is trained to handle incomplete data!") 