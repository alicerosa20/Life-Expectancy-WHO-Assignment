import streamlit as st
import joblib
import numpy as np

# Load Life Expectancy model and scaler
life_exp_model = joblib.load('models/best_life_expectancy_model.pkl')
life_exp_scaler = joblib.load('models/life_expectancy_scaler.pkl')

st.title("Life Expectancy Prediction using ML")


st.image(
    "https://assets.technologynetworks.com/production/dynamic/images/content/396360/life-expectancy-growth-has-slowed-across-europe-396360-1280x720.webp?cb=13222683",
    caption="Human Life Expectancy",
    width=250,  # smaller image size
    use_container_width=False
)

# Input fields
BMI = st.number_input("BMI (Average Body Mass Index of entire population)", 0.0, 80.0)
thinness_1_19_years = st.number_input("Prevalence of thinness among children and adolescents for Age 10 to 19 (%)", 0.0, 100.0)
thinness_5_9_years = st.number_input("Prevalence of thinness among children for Age 5 to 9(%)", 0.0, 100.0)
Alcohol = st.number_input("Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)", 0.0, 18.0)
Diphtheria = st.number_input("Diphtheria immunization coverage among 1-year-olds (%)", 0.0, 100.0)
Hepatitis_B = st.number_input("Hepatitis B (HepB) immunization coverage among 1-year-olds (%)", 0.0, 100.0)
Income_composition_of_resources = st.number_input("Income Composition of Resources", 0.0, 1.0)
Polio = st.number_input("Polio (Pol3) immunization coverage among 1-year-olds (%)", 0.0, 100.0)
Schooling = st.number_input("Schooling (years)", 0.0, 23.0)
Total_expenditure = st.number_input("General government expenditure on health as a percentage of total government expenditure (%)", 0.0, 15.0)
Adult_Mortality = st.number_input("Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)", 0.0, 800.0)
Infant_deaths = st.number_input("Number of Infant Deaths per 1000 population", 0.0, 1600.0)
Percentage_expenditure = st.number_input("Expenditure on health as a percentage of Gross Domestic Product per capita(%)", 0.0, 18000.0)
Measles = st.number_input("Measles (number of reported cases per 1000 population)", 0.0, 120000.0)
HIV_AIDS = st.number_input("Deaths per 1 000 live births HIV/AIDS (0-4 years)", 0.0, 50.0)
GDP = st.number_input("Gross Domestic Product per capita (in USD)", 0.0, 120000.0)
Population = st.number_input("Population", 0.0, 1200000000.0)
Status = st.selectbox("Status (Developing, Developed)", ["Developing", "Developed"])

# Prediction button
if st.button("Life Expectancy"):
    if Status == "Developing":
        Status = 1
    else:
        Status = 0
    life_expectancy_input = [[BMI, thinness_1_19_years, thinness_5_9_years, Alcohol, Diphtheria, Hepatitis_B,
                              Income_composition_of_resources, Polio, Schooling, Total_expenditure,
                              Adult_Mortality, Infant_deaths, Percentage_expenditure, Measles, HIV_AIDS, GDP, Population, Status]]
    scaled = life_exp_scaler.transform(life_expectancy_input)
    result = life_exp_model.predict(scaled)
    st.success(f"Predicted Life Expectancy: {result[0]:.2f} years")