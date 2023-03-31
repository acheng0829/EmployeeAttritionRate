import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import predict_page 

import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("Attrition Data.csv")
    df = df[["Age", "Attrition", "Department", "YearsAtCompany", "MaritalStatus", "EducationField", "MonthlyIncome", "NumCompaniesWorked"]]

    return df

df = load_data()

#transforms data strings to int using Label Encoder
def transform_data():
    data = predict_page.load_model()
    le_education = data["le_education"]
    le_department = data["le_department"]
    le_marital = data["le_marital"]
    le_attrition = data["le_attrition"]

    #Drop columns which are strings
    transformed_df = df.drop(columns = ['Attrition', 'Department', 'EducationField', 'MaritalStatus'])
    transformed_df = add_transformed_columns(transformed_df, le_education, le_department, le_marital, le_attrition)

    return transformed_df

def show_explore_page():
    st.title("Employee Attrition Rates Dataset Visualization")
    st.write(
        """Visualizing Data Correlation
        """
    )
    t_data = transform_data()

    show_heatmap(t_data)
    show_rate_by_age(t_data)
    show_rate_by_marital(t_data)
    show_income_by_age()
    show_pie_ed_field()
    


 


def add_transformed_columns(transformed_df, le_education, le_department, le_marital, le_attrition):
    transformed_attrition = le_attrition.fit_transform(df['Attrition'])
    transformed_df['Attrition'] = transformed_attrition.tolist()
    
    transformed_dept = le_department.fit_transform(df['Department'])
    transformed_df['Department'] = transformed_dept.tolist()

    transformed_ed_field = le_education.fit_transform(df['EducationField'])
    transformed_df['EducationField'] = transformed_ed_field.tolist()

    transformed_marital = le_marital.fit_transform(df['MaritalStatus'])
    transformed_df['MaritalStatus'] = transformed_marital.tolist()
    
    return transformed_df

def show_heatmap(t_data):
    fig = plt.figure(figsize=(14,14))
    sns.heatmap(t_data.corr(), annot=True, fmt = '.0%')
    st.write(fig)

def show_rate_by_age(t_data):
    st.write(
        """ 
        #### Attrition Rate based on Age
        """)
    data = t_data.groupby(["Age"])["Attrition"].mean().sort_values(ascending=True)
    st.bar_chart(data)

def show_rate_by_marital(t_data):
    st.write(
        """ 
        #### Attrition Rate based on Marital Status
        """)
    data = t_data.groupby(["MaritalStatus"])["Attrition"].mean().sort_values(ascending=True)
    st.bar_chart(data)
    st.write("KEY")
    st.write("0: Divorced")
    st.write("1: Married")
    st.write("2: Single")

def show_income_by_age():
    st.write(
        """
    #### Mean Monthly Income Based On Age
    """
    )
    data = df.groupby(["Age"])["MonthlyIncome"].mean().sort_values(ascending=True)
    st.line_chart(data)

def show_pie_ed_field():
    #pie percentage of data from education field
    data = df["EducationField"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Number of Data from different Education Fields""")
    st.pyplot(fig1)