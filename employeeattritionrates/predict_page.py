import streamlit as st
import pickle
import numpy as np


def load_model():
	with open('employeeattritionrates/saved_steps.pkl', 'rb') as file:
		data = pickle.load(file)
	return data

#Loads the Logistic Regression model, as well as Label Encoders 
data = load_model()
regressor_loaded = data["model"]
le_education = data["le_education"]
le_department = data["le_department"]
le_attrition = data["le_attrition"]
le_marital = data["le_marital"]

#Called when user selects Predict on sidebar
def show_predict_page():
	st.title("Predict Employee Attrition Rate")
	st.write("""Enter Potential Employee Information""")

	education_field, department, marital_status = populate_selectboxes()

	marital = st.selectbox("***Marital Status***", marital_status)
	ed_field = st.selectbox("***Education Field***", education_field)
	dept = st.selectbox("***Department***", department)
	yrs_former_company = st.text_input('***Years at Former Company*** (Enter a Positive Whole Number)')
	age = st.text_input('***Age*** (Enter a Positive Whole Number)')
	income = st.text_input('***Monthly Income Offered*** (Enter a Positive Whole Number)')
	companies_worked = st.text_input('***Number of Companies Worked*** (Enter a Positive Whole Number)')

	btn_clicked = st.button("Calculate Attrition Probability")
	if btn_clicked:
		predict_attrition_rate(age, dept, yrs_former_company, marital, ed_field, income, companies_worked)

		
def populate_selectboxes():
	education_field = (
		"Life Sciences",
		"Medical",
		"Marketing",
		"Technical Degree",
		"Human Resources",
		"Other",
	)
	department = (
		"Research & Development",
		"Sales",
		"Human Resources",
	)
	marital_status = (
		"Single",
		"Married",
		"Divorced",
	)
	return education_field, department, marital_status

def predict_attrition_rate(age, dept, yrs_former_company, marital, ed_field, income, companies_worked):
	x = np.array([[age, dept, yrs_former_company, marital, ed_field, income, companies_worked]])
	x[:, 1] = le_department.transform(x[:,1])
	x[:, 3] = le_marital.transform(x[:,3])
	x[:, 4] = le_education.transform(x[:,4])
	x = x.astype(float)

	attrition_probability = regressor_loaded.predict_proba(x) * 100
	st.subheader("The estimated probability of Employee Attrition is: ")
	st.subheader(str(round(attrition_probability[0][1], 2)) + "%")
