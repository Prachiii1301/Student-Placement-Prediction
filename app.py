import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load dataset
df = pd.read_csv("clean_student_placement_dataset.csv")

# use only 6 features
features = [
    'college_gpa',
    'coding_score',
    'internships',
    'projects',
    'technical_skills',
    'communication_skills'
]

X = df[features]
y = df['placed']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "placement_model.pkl")

print("Model saved successfully")

import streamlit as st
import numpy as np
import joblib

# load model
model = joblib.load("placement_model.pkl")

st.set_page_config(page_title="Placement Predictor", layout="wide")

# Title
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🎓 Student Placement Prediction System</h1>",
    unsafe_allow_html=True
)

st.write("### Enter student details to predict placement chances")

# Sidebar Inputs
st.sidebar.header("Student Information")

gpa = st.sidebar.slider("College GPA",0.0,10.0,7.0)
coding = st.sidebar.slider("Coding Score",0,100,60)
internships = st.sidebar.number_input("Internships",0,10,1)
projects = st.sidebar.number_input("Projects",0,10,2)
technical = st.sidebar.slider("Technical Skills",0,100,60)
communication = st.sidebar.slider("Communication Skills",0,100,60)

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Inputs")

    st.write("**College GPA:**", gpa)
    st.write("**Coding Score:**", coding)
    st.write("**Internships:**", internships)
    st.write("**Projects:**", projects)
    st.write("**Technical Skills:**", technical)
    st.write("**Communication Skills:**", communication)

with col2:
    st.subheader("Prediction")

    if st.button("Predict Placement 🚀"):

        input_data = np.array([[gpa,coding,internships,projects,technical,communication]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success("✅ Student is Likely to be Placed")
        else:
            st.error("❌ Student is Not Likely to be Placed")

        st.write("### Placement Probability")
        st.progress(int(probability*100))

        st.metric("Chance of Placement", f"{probability*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("📊 Machine Learning model predicts placement based on academic and skill features.")