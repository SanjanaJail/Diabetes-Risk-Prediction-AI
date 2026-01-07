import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



def load_and_preprocess_data():
    """Load and preprocess the realistic dataset for training"""
    print("📊 Loading and preprocessing realistic diabetes data...")
    
   
    csv_path =r"C:\Users\sanja\Downloads\diabetes_realistic_missing_data_consistent.csv"
    df = pd.read_csv(csv_path)
    
   
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values per column (%):")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(missing_percent.round(2))
    
   
    feature_columns = [
        'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 
        'BP_Systolic', 'BP_Diastolic', 'Cholesterol', 'Stress', 
        'Hereditary', 'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose', 
        'C_Peptide', 'Ketones', 'Antibodies'  
    ]
    
    
    X = df[feature_columns].copy()
    y_stage = df['Diabetes_Stage'].copy()  #  No Diabetes, Pre-Diabetes, Diabetes
    y_type = df['Diabetes_Type'].copy()    #  None, Type 1, Type 2
    
   
    print(" Preprocessing categorical features...")
    
   
    X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1, np.nan: np.nan})
    
  
    stress_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, np.nan: np.nan}
    X['Stress'] = X['Stress'].map(stress_mapping)
    
   
    print(" Handling missing values with sophisticated imputation...")
    
   
    print(" Creating missingness indicator features...")
    for col in feature_columns:
        if col not in ['Stress', 'Gender']:  
            flag_col = f"{col}_missing"
            X[flag_col] = X[col].isnull().astype(int)
    
   
    extended_feature_columns = feature_columns + [f"{col}_missing" for col in feature_columns if col not in ['Stress', 'Gender']]
    
   
    lab_columns = ['HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose', 'C_Peptide', 'Ketones']
    lab_imputer = SimpleImputer(strategy='mean')
    X[lab_columns] = lab_imputer.fit_transform(X[lab_columns])
    
   
    binary_columns = ['Antibodies', 'Hereditary']
    binary_imputer = SimpleImputer(strategy='most_frequent')
    X[binary_columns] = binary_imputer.fit_transform(X[binary_columns])
    
  
    other_num_columns = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 'Cholesterol']
    other_imputer = SimpleImputer(strategy='mean')
    X[other_num_columns] = other_imputer.fit_transform(X[other_num_columns])
    
    
    categorical_columns = ['Gender', 'Stress']
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])
    
   
    le_stage = LabelEncoder()
    le_type = LabelEncoder()
    
    y_stage_encoded = le_stage.fit_transform(y_stage)
    y_type_encoded = le_type.fit_transform(y_type)
    

    stage_mapping = dict(zip(le_stage.classes_, range(len(le_stage.classes_))))
    type_mapping = dict(zip(le_type.classes_, range(len(le_type.classes_))))
    
    print(f"Stage mapping: {stage_mapping}")
    print(f"Type mapping: {type_mapping}")
    
    return X, y_stage_encoded, y_type_encoded, le_stage, le_type, extended_feature_columns

def train_stage_model(X, y_stage, feature_columns):
    """Train the Diabetes Stage classifier (3-class: No, Pre, Diabetes)"""
    print("\n" + "="*50)
    print(" TRAINING DIABETES STAGE MODEL (3-class)")
    print("="*50)
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_stage, test_size=0.2, random_state=42, stratify=y_stage
    )
    
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    print(" Tuning hyperparameters for Stage Model...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_stage_model = grid_search.best_estimator_
    print(f" Best parameters: {grid_search.best_params_}")
    
    # Cross-validation score
    cv_scores = cross_val_score(best_stage_model, X_train_scaled, y_train, cv=5)
    print(f" Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Evaluate on test set
    y_pred = best_stage_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Test Accuracy: {accuracy:.4f}")
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Diabetes', 'No Diabetes', 'Pre-Diabetes']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_stage_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n Top 10 Most Important Features for Stage Prediction:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Features - Diabetes Stage Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('stage_feature_importance.png')
    plt.close()
    
    return best_stage_model, scaler, accuracy

def train_type_model(X, y_type, feature_columns):
    """Train the Diabetes Type classifier (only on Diabetes cases - 2-class: Type 1 vs Type 2)"""
    print("\n" + "="*50)
    print("TRAINING DIABETES TYPE MODEL (2-class)")
    print("="*50)
    

    diabetes_mask = (y_type != 0)
    X_diabetes = X[diabetes_mask]
    y_diabetes = y_type[diabetes_mask]
    
    print(f" Diabetes cases for type training: {len(X_diabetes)}")
    
    if len(X_diabetes) < 100:
        print(" Not enough diabetes cases found for type training!")
        return None, None, 0
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
    )
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
   
    print(" Tuning hyperparameters for Type Model...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
   
    best_type_model = grid_search.best_estimator_
    print(f" Best parameters: {grid_search.best_params_}")
    
   
    cv_scores = cross_val_score(best_type_model, X_train_scaled, y_train, cv=5)
    print(f" Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
   
    y_pred = best_type_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Test Accuracy: {accuracy:.4f}")
    
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Type 1', 'Type 2']))
    
   
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_type_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n Top 10 Most Important Features for Type Prediction:")
    print(feature_importance.head(10))
    
    
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Features - Diabetes Type Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('type_feature_importance.png')
    plt.close()
    
    return best_type_model, scaler, accuracy

def main():
    """Main training function"""
    print("Starting Advanced Diabetes Model Training...")
    print("="*60)
    
    
    X, y_stage, y_type, le_stage, le_type, feature_columns = load_and_preprocess_data()
    
   
    stage_model, stage_scaler, stage_accuracy = train_stage_model(X, y_stage, feature_columns)
    
   
    type_model, type_scaler, type_accuracy = train_type_model(X, y_type, feature_columns)
    
  
    print("\n Saving models and metadata...")
    
   
    joblib.dump(stage_model, 'diabetes_stage_model.pkl')
    joblib.dump(stage_scaler, 'stage_scaler.pkl')
    joblib.dump(le_stage, 'stage_label_encoder.pkl')
    
    
    if type_model is not None:
        joblib.dump(type_model, 'diabetes_type_model.pkl')
        joblib.dump(type_scaler, 'type_scaler.pkl')
        joblib.dump(le_type, 'type_label_encoder.pkl')
    
    
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    
    metadata = {
        'stage_accuracy': stage_accuracy,
        'type_accuracy': type_accuracy if type_model else None,
        'dataset_size': len(X),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_columns': feature_columns,
        'stage_classes': list(le_stage.classes_),
        'type_classes': list(le_type.classes_)
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    
    print(" All models and metadata saved successfully!")
    
    
    print("\n" + "="*60)
    print(" ADVANCED TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f" Diabetes Stage Model Accuracy: {stage_accuracy:.4f}")
    if type_model:
        print(f" Diabetes Type Model Accuracy: {type_accuracy:.4f}")
    print(f" Total Training Samples: {len(X)}")
    print(f" Features used: {len(feature_columns)} (including missingness flags)")
    print(f" Models saved:")
    print("   - diabetes_stage_model.pkl")
    print("   - diabetes_type_model.pkl")
    print("   - stage_scaler.pkl, type_scaler.pkl")
    print("   - stage_label_encoder.pkl, type_label_encoder.pkl")
    print("   - feature_columns.pkl")
    print("   - model_metadata.pkl")
    print("   - stage_feature_importance.png")
    print("   - type_feature_importance.png")

if __name__ == "__main__":
    main()