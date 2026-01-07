# diabetes_predictor_complete.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        """
        Diabetes Predictor - Automatically trains model if not found
        """
        self.model_file = 'diabetes_prediction_model.pkl'
        self.scaler_file = 'scaler.pkl'
        self.data_file = r"C:\Users\sanja\Downloads\diabetes.csv"
        
        # Check if model exists, otherwise train it
        if self._model_exists():
            print("✅ Loading existing model...")
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)
        else:
            print("🔄 Training new model...")
            self._train_and_save_model()
    
    def _model_exists(self):
        """Check if model files exist"""
        return os.path.exists(self.model_file) and os.path.exists(self.scaler_file)
    
    def _train_and_save_model(self):
        """Train and save the model"""
        # Load data
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_file)
        print(f"   Loaded {len(df)} patient records")
        
        # Data preprocessing
        print("🔧 Preprocessing data...")
        columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_fix:
            zero_count = (df[column] == 0).sum()
            if zero_count > 0:
                median_val = df[column].median()
                df[column] = df[column].replace(0, median_val)
                print(f"   Fixed {column}: Replaced {zero_count} zeros with median {median_val:.2f}")
        
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        print(f"   Diabetes cases: {y.sum()} ({y.mean()*100:.1f}% of dataset)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Testing Accuracy: {test_accuracy:.4f}")
        
        # Save model
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)
        print("💾 Model saved successfully!")
        
        # Show feature importance
        print("\n📈 Top 5 Most Important Features:")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, row in feature_importance.head().iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    def predict_risk(self, patient_data):
        """
        Predict diabetes risk for a new patient
        
        Parameters:
        patient_data (dict): Dictionary with these features:
            - Pregnancies
            - Glucose  
            - BloodPressure
            - SkinThickness
            - Insulin
            - BMI
            - DiabetesPedigreeFunction
            - Age
        """
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Scale features
            patient_scaled = self.scaler.transform(patient_df)
            
            # Make prediction
            probability = self.model.predict_proba(patient_scaled)[0][1]
            prediction = self.model.predict(patient_scaled)[0]
            
            # Get risk level
            if probability < 0.3:
                risk_level = "LOW"
            elif probability < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            return {
                'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
                'probability': round(probability, 4),
                'risk_percentage': round(probability * 100, 2),
                'risk_level': risk_level,
                'confidence': 'High' if probability > 0.8 or probability < 0.2 else 'Medium'
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None

def get_user_input():
    """Get patient data from user input"""
    print("\n📝 ENTER PATIENT INFORMATION")
    print("=" * 40)
    
    patient_data = {}
    
    # Get each feature from user
    patient_data['Pregnancies'] = int(input("Number of pregnancies: ") or 0)
    patient_data['Glucose'] = float(input("Glucose level (mg/dL): ") or 100)
    patient_data['BloodPressure'] = float(input("Blood pressure (mm Hg): ") or 70)
    patient_data['SkinThickness'] = float(input("Skin thickness (mm): ") or 20)
    patient_data['Insulin'] = float(input("Insulin level (mu U/ml): ") or 80)
    patient_data['BMI'] = float(input("BMI (kg/m²): ") or 25.0)
    patient_data['DiabetesPedigreeFunction'] = float(input("Diabetes pedigree function: ") or 0.5)
    patient_data['Age'] = int(input("Age (years): ") or 30)
    
    return patient_data

def show_sample_values():
    """Show typical values for reference"""
    print("\n💡 TYPICAL VALUE RANGES:")
    print("   - Glucose: Normal < 100, Pre-diabetes 100-125, Diabetes ≥ 126")
    print("   - Blood Pressure: Normal < 120/80, High ≥ 140/90")
    print("   - BMI: Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥ 30")
    print("   - Skin Thickness: Typical 20-40 mm")
    print("   - Insulin: Fasting 2-25 μU/mL")

def main():
    print("=" * 50)
    print("🩺 DIABETES RISK PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize predictor (will auto-train if needed)
    predictor = DiabetesPredictor()
    
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Predict diabetes risk for a new patient")
        print("2. View sample predictions")
        print("3. Show typical value ranges")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Get user input for patient data
            show_sample_values()
            patient_data = get_user_input()
            
            print("\n" + "=" * 40)
            print("PATIENT DATA SUMMARY:")
            print("=" * 40)
            for key, value in patient_data.items():
                print(f"  {key}: {value}")
            
            # Make prediction
            result = predictor.predict_risk(patient_data)
            
            if result:
                print("\n🔍 PREDICTION RESULTS:")
                print("=" * 40)
                print(f"   Diagnosis: {result['prediction']}")
                print(f"   Risk Level: {result['risk_level']}")
                print(f"   Probability: {result['probability']}")
                print(f"   Risk Percentage: {result['risk_percentage']}%")
                print(f"   Confidence: {result['confidence']}")
                
                # Additional advice based on risk
                print(f"\n💡 RECOMMENDATION:")
                if result['risk_level'] == 'LOW':
                    print("   Maintain healthy lifestyle with regular checkups")
                elif result['risk_level'] == 'MEDIUM':
                    print("   Monitor regularly and consider lifestyle changes")
                else:
                    print("   Consult a healthcare professional for further evaluation")
            
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            # Show sample predictions
            print("\n" + "=" * 50)
            print("🧪 SAMPLE PREDICTIONS")
            print("=" * 50)
            
            sample_patients = [
                {
                    'Pregnancies': 2,
                    'Glucose': 140,  # High glucose
                    'BloodPressure': 70,
                    'SkinThickness': 30,
                    'Insulin': 80,
                    'BMI': 33.0,  # High BMI
                    'DiabetesPedigreeFunction': 0.5,
                    'Age': 35
                },
                {
                    'Pregnancies': 1,
                    'Glucose': 90,  # Normal glucose
                    'BloodPressure': 70,
                    'SkinThickness': 25,
                    'Insulin': 60,
                    'BMI': 24.0,  # Normal BMI
                    'DiabetesPedigreeFunction': 0.2,
                    'Age': 25
                }
            ]
            
            for i, patient in enumerate(sample_patients, 1):
                print(f"\n--- Sample Patient {i} ---")
                print("Data:", patient)
                
                result = predictor.predict_risk(patient)
                if result:
                    print(f"Result: {result['prediction']} (Risk: {result['risk_level']}, {result['risk_percentage']}%)")
            
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            show_sample_values()
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            print("\n👋 Thank you for using Diabetes Risk Prediction System!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()