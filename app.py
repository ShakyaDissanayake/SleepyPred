import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="💤",
    layout="wide"
)

def load_model_and_scaler():
    """Load the saved model and scaler"""
    try:
        with open('sleep_disorder_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('sleep_disorder_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure both files exist in the current directory.")
        return None, None

def preprocess_input(data):
    """Preprocess the input data"""
    # Expected columns after one-hot encoding
    expected_columns = [
        'Gender', 'Age', 'Quality of Sleep', 'Physical Activity Level',
        'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps',
        'BP Category', 'Sleep Category', 'Occupation_Doctor',
        'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Manager',
        'Occupation_Nurse', 'Occupation_Sales Representative',
        'Occupation_Salesperson', 'Occupation_Scientist',
        'Occupation_Software Engineer', 'Occupation_Teacher'
    ]
    
    # Convert Gender to binary
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Convert Sleep Category to binary
    data['Sleep Category'] = data['Sleep Category'].map({'Short Sleep': 0, 'Optimal Sleep': 1})
    
    # Convert BMI Category to ordinal
    bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    data['BMI Category'] = data['BMI Category'].map(bmi_mapping)
    
    # Convert BP Category to ordinal
    bp_mapping = {
        'Optimal Blood Pressure': 0,
        'Elevated Blood Pressure': 1,
        'Stage 1 Hypertension': 2,
        'Stage 2 Hypertension': 3
    }
    data['BP Category'] = data['BP Category'].map(bp_mapping)
    
    # One-hot encode Occupation
    occupation_columns = [col for col in expected_columns if col.startswith('Occupation_')]
    for col in occupation_columns:
        occupation = col.replace('Occupation_', '')
        data[col] = (data['Occupation'] == occupation).astype(int)
    
    # Drop original Occupation column
    data = data.drop('Occupation', axis=1)
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
            
    # Reorder columns to match training data
    data = data[expected_columns]
    
    return data

def predict_sleep_disorder(data, model, scaler):
    """Make predictions using the loaded model"""
    # Scale the features
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)
    
    # Map predictions to labels
    sleep_disorder_mapping = {0: 'None', 1: 'Insomnia', 2: 'Sleep Apnea'}
    predicted_disorder = sleep_disorder_mapping[prediction[0]]
    
    # Create probability dictionary
    prob_dict = {
        'None': probabilities[0][0],
        'Insomnia': probabilities[0][1],
        'Sleep Apnea': probabilities[0][2]
    }
    
    return predicted_disorder, prob_dict

def main():
    st.title("💤 Sleep Disorder Prediction App 💤")
    st.write("""
    This app predicts the type of sleep disorder a person might have based on their health and lifestyle data.
    Fill in the form below to get a prediction.
    """)
    
    # Medical Disclaimer
    st.warning("""
    ⚕️ **MEDICAL DISCLAIMER**: This application is for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. 
    The predictions provided are based on statistical models and should not be considered as medical diagnoses. 
    If you suspect you have a sleep disorder or are experiencing sleep-related issues, please consult a qualified healthcare professional for proper evaluation and treatment.
    """)
    
    # Create a form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("👤 Gender", options=["Male", "Female"])
            age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
            sleep_quality = st.slider("😴 Quality of Sleep (1-10)", min_value=1, max_value=10, value=7)
            physical_activity = st.slider("🏃‍♂️ Physical Activity Level (minutes per day)", min_value=0, max_value=120, value=60)
            stress_level = st.slider("😰 Stress Level (1-10)", min_value=1, max_value=10, value=5)
        
        with col2:
            bmi_category = st.selectbox("⚖️ BMI Category", options=["Underweight", "Normal", "Overweight", "Obese"])
            heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            daily_steps = st.number_input("👣 Daily Steps", min_value=0, max_value=30000, value=8000)
            bp_category = st.selectbox("🩸 Blood Pressure Category", options=[
                "Optimal Blood Pressure", 
                "Elevated Blood Pressure", 
                "Stage 1 Hypertension", 
                "Stage 2 Hypertension"
            ])
            sleep_category = st.selectbox("⏰ Sleep Duration Category", options=["Short Sleep", "Optimal Sleep"])
            
        occupation = st.selectbox("💼 Occupation", options=[
            "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", 
            "Sales Representative", "Salesperson", "Scientist", 
            "Software Engineer", "Teacher"
        ])
        
        submit_button = st.form_submit_button("🔍 Predict Sleep Disorder")
    
    # When the form is submitted
    if submit_button:
        # Load model and scaler
        model, scaler = load_model_and_scaler()
        
        if model is not None and scaler is not None:
            # Create a dataframe with the user input
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Quality of Sleep': [sleep_quality],
                'Physical Activity Level': [physical_activity],
                'Stress Level': [stress_level],
                'BMI Category': [bmi_category],
                'Heart Rate': [heart_rate],
                'Daily Steps': [daily_steps],
                'BP Category': [bp_category],
                'Sleep Category': [sleep_category],
                'Occupation': [occupation]
            })
            
            # Preprocess input data
            processed_data = preprocess_input(input_data)
            
            # Make prediction
            predicted_disorder, probabilities = predict_sleep_disorder(processed_data, model, scaler)
            
            # Display results
            st.header("🔮 Prediction Results")
            
            # Show the predicted disorder with different colors based on severity
            if predicted_disorder == "None":
                st.success(f"Predicted Sleep Disorder: ✅ {predicted_disorder}")
            elif predicted_disorder == "Insomnia":
                st.warning(f"Predicted Sleep Disorder: ⚠️ {predicted_disorder}")
            else:
                st.error(f"Predicted Sleep Disorder: 🚨 {predicted_disorder}")
            
            # Display probabilities as a bar chart
            st.subheader("📊 Disorder Probabilities")
            prob_df = pd.DataFrame({
                'Disorder': list(probabilities.keys()),
                'Probability': list(probabilities.values())
            })
            st.bar_chart(prob_df.set_index('Disorder'))
            
            # Display additional information based on the prediction
            st.subheader("ℹ️ What does this mean?")
            if predicted_disorder == "None":
                st.write("""
                ✨ Good news! Based on your inputs, our model predicts you don't have a sleep disorder.
                However, it's still important to maintain good sleep hygiene and healthy habits.
                
                🩺 If you're still experiencing sleep issues despite this prediction, please consult with a healthcare provider for a thorough evaluation.
                """)
            elif predicted_disorder == "Insomnia":
                st.write("""
                😴 Our model suggests you might be experiencing insomnia. Insomnia is characterized by difficulty
                falling asleep, staying asleep, or getting good quality sleep.
                
                🩺 **Important:** If you think you have insomnia, please consult with a sleep specialist or healthcare provider for proper diagnosis and treatment options. Effective treatments are available, including cognitive behavioral therapy for insomnia (CBT-I) and other approaches.
                """)
            else:  # Sleep Apnea
                st.write("""
                😷 Our model suggests you might be experiencing sleep apnea. Sleep apnea is a serious sleep disorder
                where breathing repeatedly stops and starts during sleep. Common symptoms include loud snoring,
                gasping for air during sleep, and feeling tired even after a full night's sleep.
                
                🩺 **Important:** If you think you have sleep apnea, please consult with a sleep specialist or healthcare provider immediately for proper diagnosis and treatment. Sleep apnea is a serious condition that requires medical attention and can be treated with CPAP therapy, oral appliances, or other interventions.
                """)
            
            # Additional disclaimer after prediction
            st.info("""
            ⚠️ **REMINDER:** This prediction is not a medical diagnosis. The accuracy of this prediction depends on the quality of input data and the limitations of the underlying model. Always seek professional medical advice for sleep-related concerns.
            """)
    
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This app uses a machine learning model to predict sleep disorders based on various health and lifestyle factors.
    
    **Note:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)
    
    st.sidebar.header("🔍 Features Used for Prediction")
    st.sidebar.write("""
    - 👤 Gender
    - 🎂 Age
    - 😴 Quality of Sleep
    - 🏃‍♂️ Physical Activity Level
    - 😰 Stress Level
    - ⚖️ BMI Category
    - ❤️ Heart Rate
    - 👣 Daily Steps
    - 🩸 Blood Pressure Category
    - ⏰ Sleep Duration Category
    - 💼 Occupation
    """)
    
    # Additional medical information in sidebar
    st.sidebar.header("🏥 When to See a Doctor")
    st.sidebar.write("""
    Consider consulting a healthcare provider if you experience:
    
    - Persistent difficulty falling or staying asleep
    - Loud snoring with gasping or choking sounds
    - Excessive daytime sleepiness
    - Morning headaches
    - Difficulty concentrating during the day
    - Irritability or mood changes related to sleep
    - Waking up feeling unrefreshed
    """)

if __name__ == "__main__":
    main()