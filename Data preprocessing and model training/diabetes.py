import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder

csv_path = r"C:\Users\sanja\Downloads\3_binary_outcomes_diabetes.csv"
data = pd.read_csv(csv_path)


print(data.head())
print("\nData shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())


le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Stress'] = le.fit_transform(data['Stress'])


data['Diabetes_Status'] = data.apply(lambda x: 'Diabetes' if x['Diabetes_Outcome'] == 1 
                                    else 'Pre-Diabetes' if x['PreDiabetes_Outcome'] == 1 
                                    else 'No Diabetes', axis=1)


data = data.drop(columns=['NoDiabetes_Outcome', 'PreDiabetes_Outcome', 'Diabetes_Outcome'])


features = ['Age', 'Gender', 'BMI', 'Cholesterol', 'Stress', 'Hereditary', 
            'HbA1c', 'Fasting_Glucose', 'Post_Meal_Glucose']

target = 'Diabetes_Status'


X = data[features]
Y = data[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


log_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
log_model.fit(X_train, Y_train)
Y_pred_log = log_model.predict(X_test)


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)


print("\nLogistic Regression Accuracy:", accuracy_score(Y_test, Y_pred_log))
print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))
print("\nLogistic Regression Report:\n", classification_report(Y_test, Y_pred_log))
print("Random Forest Report:\n", classification_report(Y_test, Y_pred_rf))
print("\nLogistic Regression CM:\n", confusion_matrix(Y_test, Y_pred_log))
print("Random Forest CM:\n", confusion_matrix(Y_test, Y_pred_rf))


feature_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_imp)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Feature Importance')
plt.show()


joblib.dump(rf_model, "diabetes_model1.pkl")
print("\n✅ Model saved as 'diabetes_model1.pkl'")
print("🔍 Model trained on features:", features)