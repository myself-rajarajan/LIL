import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from fpdf import FPDF

# Load and Clean the Cleveland Dataset
cleveland_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                "restecg", "thalach", "exang", "oldpeak", "slope", 
                "ca", "thal", "target"]
df = pd.read_csv(cleveland_url, names=column_names)

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Prepare Data
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
log_reg = LogisticRegression(random_state=42, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train_scaled, y_train)
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Save Models
joblib.dump(log_reg, "log_reg_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Feature descriptions for input
feature_descriptions = {
    "age": "Age (years)",
    "sex": "Sex (1 = Male, 0 = Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar (>120 mg/dL, 1 = True, 0 = False)",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (0-2)",
    "ca": "Number of Major Vessels (0-3)",
    "thal": "Thalassemia Type (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)"
}

# Diet Recommendations
diet_chart = [
    ["Food Type", "Recommended Items"],
    ["Fruits", "Apples, Berries, Oranges"],
    ["Vegetables", "Spinach, Carrots, Broccoli"],
    ["Proteins", "Fish, Chicken, Beans"],
    ["Grains", "Oats, Brown Rice, Whole Wheat"],
    ["Dairy", "Low-fat Milk, Yogurt"],
    ["Healthy Fats", "Olive Oil, Nuts, Avocado"]
]

# Risk level classifier
def risk_level(prob):
    if prob < 0.4:
        return "Low Risk"
    elif 0.4 <= prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

# Chart plotting function
def plot_risk_chart(prob_log, prob_xgb, prob_nb, report_filename):
    risks = [prob_log * 100, prob_xgb * 100, prob_nb * 100]
    labels = ["Logistic Regression", "XGBoost", "Naïve Bayes"]
    colors = ['green' if r < 40 else 'yellow' if r < 70 else 'red' for r in risks]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=risks, palette=colors)
    plt.ylabel("Heart Disease Probability (%)")
    plt.title("Heart Disease Risk Analysis")
    plt.ylim(0, 100)
    
    os.makedirs("reports", exist_ok=True)
    chart_path = f"reports/{report_filename}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# PDF generation
def generate_pdf_report(risk, prob_log, prob_xgb, prob_nb, user_inputs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Heart_Disease_Report_{timestamp}"
    
    chart_path = plot_risk_chart(prob_log, prob_xgb, prob_nb, report_filename)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Machine Learning-Based Heart Disease Prediction System Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Patient's Health Data:", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for key, value in user_inputs.items():
        pdf.cell(95, 8, key, border=1)
        pdf.cell(95, 8, str(value), border=1, ln=True)
    
    pdf.ln(10)
    pdf.image(chart_path, x=10, w=180)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"Predicted Risk Level: {risk}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Logistic Regression Risk: {prob_log*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"XGBoost Risk: {prob_xgb*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"Naïve Bayes Risk: {prob_nb*100:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, "Diet Plan Recommendations:", ln=True)
    pdf.ln(5)
    for row in diet_chart:
        for item in row:
            pdf.cell(95, 8, item, border=1)
        pdf.ln(8)
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", 12)
    pdf.multi_cell(0, 10, '"A healthy heart leads to a healthy life. Take care of your body, and it will take care of you."')
    
    report_path = f"reports/{report_filename}.pdf"
    pdf.output(report_path)
    print(f"\n✅ Report generated successfully: {report_path}")

# Entry point for user input and prediction
if __name__ == "__main__":
    print("\n💬 Enter your health data to generate a heart disease report.")
    user_inputs = {}
    for col, desc in feature_descriptions.items():
        while True:
            try:
                user_inputs[desc] = float(input(f"{desc}: "))
                break
            except ValueError:
                print("❌ Please enter a valid number.")

    input_df = pd.DataFrame([list(user_inputs.values())], columns=X.columns)
    input_scaled = scaler.transform(input_df)

    prob_log = log_reg.predict_proba(input_scaled)[:, 1][0]
    prob_xgb = xgb_model.predict_proba(input_scaled)[:, 1][0]
    prob_nb = nb_model.predict_proba(input_scaled)[:, 1][0]

    avg_prob = (prob_log + prob_xgb + prob_nb) / 3
    risk = risk_level(avg_prob)

    generate_pdf_report(risk, prob_log, prob_xgb, prob_nb, user_inputs)
