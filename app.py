import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_filename = "employee_performance_model.joblib"
best_model = joblib.load(model_filename)

# Load dataset to check column structure
file = 'INX_Future_Inc_Employee_Performance.csv'
df = pd.read_csv(file)

# Setting the EmpNumber as index
df.set_index('EmpNumber', inplace=True)

# Streamlit App Title
st.title("🏆 Employee Performance Prediction Model")

# Streamlit App SubTitle
st.subheader("This model will be used to predict the performance rating of employees based on specific feature inputs")

# Checkbox to show the dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

age = st.number_input("Age", min_value=18, max_value=70, value=25, step=1)
gender = st.selectbox("Gender: 0-Female, 1-Male", [0, 1])
education_background = st.selectbox(
    "EducationBackground: 0-Human Resources, 1-Life Sciences, 2-Marketing, 3-Medical, 4-Other, 5-Technical Degree",
    list(range(6))
)
marital_status = st.selectbox("MaritalStatus: 0-Divorced, 1-Married, 2-Single", [0, 1, 2])
emp_department = st.selectbox(
    "EmpDepartment: 0-Data Science, 1-Development, 2-Finance, 3-Human Resources, 4-Research & Development(R&D), 5-Sales",
    list(range(6))
)
emp_job_role = st.selectbox(
    "EmpJobRole: 0-Business Analyst, 1-Data Scientist, 2-Delivery Manager, 3-Developer, 4-Finance Manager, "
    "5-Human Resources, 6-HealthCare Representative, 7-Laboratory Technician, 8-Manager, 9-Manager R&D, "
    "10-Manufacturing Director, 11-Research Director, 12-Research Scientist, 13-Sales Executive, "
    "14-Sales Representative, 15-Senior Developer, 16-Senior Manager R&D, 17-Technical Architect, 18-Technical Lead",
    list(range(19))
)
business_travel = st.selectbox("BusinessTravelFrequency: 0-Travel_Frequently, 1-Travel_Rarely, 2-Non-Travel", [0, 1, 2])
distance_from_home = st.number_input("DistanceFromHome", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
emp_education_level = st.selectbox("EmpEducationLevel: 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor", [1, 2, 3, 4, 5])
emp_environment_satisfaction = st.selectbox("EmpEnvironmentSatisfaction: 1-Low, 2-Medium, 3-High, 4-Very High", [1, 2, 3, 4])
emp_hourly_rate = st.number_input("EmpHourlyRate", min_value=0, value=25, step=1)
emp_job_involvement = st.selectbox("EmpJobInvolvement: 1-Low, 2-Medium, 3-High, 4-Very High", [1, 2, 3, 4])
emp_job_level = st.number_input("EmpJobLevel", min_value=0, value=2, step=1)
emp_job_satisfaction = st.selectbox("EmpJobSatisfaction: 1-Low, 2-Medium, 3-High, 4-Very High", [1, 2, 3, 4])
num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, value=2, step=1)
over_time = st.selectbox("OverTime: 0-No, 1-Yes", [0, 1])
emp_last_salary_hike = st.number_input("EmpLastSalaryHikePercent", min_value=0.0, value=10.0, step=0.1)
emp_relationship_satisfaction = st.selectbox("EmpRelationshipSatisfaction: 1-Low, 2-Medium, 3-High, 4-Very High", [1, 2, 3, 4])
total_work_experience = st.number_input("TotalWorkExperienceInYears", min_value=0, value=5, step=1)
training_times_last_year = st.number_input("TrainingTimesLastYear", min_value=0, value=2, step=1)
emp_work_life_balance = st.selectbox("EmpWorkLifeBalance: 1-Bad, 2-Good, 3-Better, 4-Best", [1, 2, 3, 4])
experience_years_at_company = st.number_input("ExperienceYearsAtThisCompany", min_value=0, value=3, step=1)
experience_years_in_role = st.number_input("ExperienceYearsInCurrentRole", min_value=0, value=2, step=1)
years_since_last_promotion = st.number_input("YearsSinceLastPromotion", min_value=0, value=1, step=1)
years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, value=2, step=1)
attrition = st.selectbox("Attrition: 0-No, 1-Yes", [0, 1])

# Prepare input data for prediction
input_data = pd.DataFrame([[age, gender, education_background, marital_status, emp_department, emp_job_role, 
                            business_travel, distance_from_home, emp_education_level, emp_environment_satisfaction,
                            emp_hourly_rate, emp_job_involvement, emp_job_level, emp_job_satisfaction, num_companies_worked,
                            over_time, emp_last_salary_hike, emp_relationship_satisfaction, total_work_experience,
                            training_times_last_year, emp_work_life_balance, experience_years_at_company,
                            experience_years_in_role, years_since_last_promotion, years_with_curr_manager, attrition]],
                          )


# Predict Performance
if st.button("Predict Performance"):
    prediction = best_model.predict(input_data)

    # Reverse label shift to match original performance ratings
    predictions = prediction + 2  
    performance_map = {1: "Low(1)", 2: "Good(2)", 3: "Excellent(3)", 4: "Outstanding(4)"}
    predicted_performance = performance_map.get(predictions[0], "Unknown")
    
    st.success(f"🎯 Predicted Employee Performance Rating: **{predicted_performance}**")

    # 🎈 Show Balloons when prediction is successful
    st.balloons()
