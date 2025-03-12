import streamlit as st
import pandas as pd
import joblib
import numpy as np


pipeline = joblib.load('Pipeline.pkl')
work_le = joblib.load('Work_LabelEncoder.pkl')
education_le = joblib.load('Education_LabelEncoder.pkl')
marital_le = joblib.load('Marital_LabeEncoder.pkl')
occupation_le = joblib.load('Occupation_LabelEncoder.pkl')
relation_le = joblib.load('Relationship_LabelEncoder.pkl')
race_le = joblib.load('Race_LabelEncoder.pkl')
sex_le = joblib.load('Sex_LabelEncoder.pkl')
country = joblib.load('Country_LabelEncoder.pkl')
model = joblib.load('Decision_tree.pkl')




st.title('Prediction App')
# st.write(pipeline.feature_names_in_)

st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            color: #333333;
        }
        [data-testid="stApp"] 
            { background-image: url("https://wallpaperaccess.com/full/825782.jpg");
            background-color: white;
            # background-image: url(https://wallpaperaccess.com/full/825782.jpg);
            background-size: 100% 100%;
            background-blend-mode: overlay;
            background-color: rgba(1, 1, 1, 0.002); /* Adjust transparency */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: white;
            text-align: center;
        }
        /* Style button */
        .stButton>button {
            background-color: black !important;  /* Default black background */
            color: white !important;            /* White text */
            font-size: 18px;
            border-radius: 5px;
            padding: 8px 20px;
            border: none;
            transition: background-color 0.3s ease;
        }
        /* Keep button black even on hover */
        .stButton>button:hover {
            background-color: black !important; /* Remains black */
            color: white !important;           /* Text stays white */
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)



cols,cols1 = st.columns(2)
with cols:
    age = st.number_input('Enter your Age',min_value = 0,step = 1)
    workclass = st.selectbox("Work Class",[' State-gov',' Self-emp-not-inc',' Private',' Federal-gov',' Local-gov',
                                        ' Self-emp-inc',' Without-pay'],index = None)
    education = st.selectbox('Education',[' Bachelors',' HS-grad',' 11th',' Masters',' 9th',' Some-college',' Assoc-acdm',' Assoc-voc',
                                        ' 7th-8th',' Doctorate',' Prof-school',' 5th-6th',' 10th',' Preschool',' 12th',' 1st-4th'],index = None)
    marital_status = st.selectbox('Marital Status',[' Never-married',' Married-civ-spouse',' Divorced',
                                                    ' Married-spouse-absent',' Separated',' Married-AF-spouse',' Widowed'],index = None)
    occupation = st.selectbox('Occupation',[' Adm-clerical',' Exec-managerial',' Handlers-cleaners',' Prof-specialty',' Other-service',
                                            ' Sales',' Craft-repair',' Transport-moving',' Farming-fishing',' Machine-op-inspct',
                                            ' Tech-support',' Protective-serv',' Armed-Forces',' Priv-house-serv'],index = None)
    relationship = st.selectbox('Relationship',[' Not-in-family',' Husband',' Wife',' Own-child',' Unmarried',' Other-relative'],index = None)

with cols1:
    race = st.selectbox('Race',[' White',' Black',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other'],index = None)
    sex = st.selectbox('Sex',[' Male',' Female'],index = None)
    capital_gain = st.number_input('income from capital investments',min_value = 0,step = 1)
    capital_loss = st.number_input('Losses from capital investments',min_value = 0,step = 1)
    hours_per_week = st.number_input('Number of hours worked per week',min_value = 0,step = 1)
    native_country = st.selectbox('Country',[' United-States',' Cuba',' Jamaica',' India',' Mexico',' Puerto-Rico',' Honduras',
                                             ' England',' Canada',' Germany',' Iran',' Philippines',' Poland',' Columbia',' Cambodia',
                                             ' Thailand',' Ecuador',' Laos',' Taiwan',' Haiti',' Portugal',' Dominican-Republic',
                                             ' El-Salvador',' France',' Guatemala',' Italy',' China',' South',' Japan',' Yugoslavia',
                                             ' Peru',' Outlying-US(Guam-USVI-etc)',' Scotland',' Trinadad&Tobago',' Greece',
                                             ' Nicaragua',' Vietnam',' Hong',' Ireland',' Hungary',' Holand-Netherlands'],index = None)

if st.button('predict'):
    workc_encoder = work_le.transform([workclass])[0]
    education_encoder = education_le.transform([education])[0]
    marital_encoder = marital_le.transform([marital_status])[0]
    occupation_encoder = occupation_le.transform([occupation])[0]
    relationship_encoder = relation_le.transform([relationship])[0]
    race_encoder = race_le.transform([race])[0]
    sex_encoder = sex_le.transform([sex])[0]
    country_encoder = country.transform([native_country])[0]

    user_data = np.array([[age,workc_encoder,education_encoder,marital_encoder,occupation_encoder,relationship_encoder,
                        race_encoder,sex_encoder,capital_gain,capital_loss,hours_per_week,country_encoder]])

    predict = model.predict(user_data)
    prediction = predict[0]
    if prediction == 1:
        st.markdown("""
            <style>
                .custom-text {
                    font-size: 20px; /* Adjust text size */
                    font-weight: bold; /* Make text bold */
                    color: white !important; /* Force font color */
                    font-family: 'Times New Roman', serif; /* Set font style */
                    text-align: center; /* Center align text */
                }
            </style>
            <p class="custom-text">Your Income is Greater Than or Equal to 50K</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .custom-text {
                    font-size: 20px; /* Adjust text size */
                    font-weight: bold; /* Make text bold */
                    color: white !important; /* Force font color */
                    font-family: 'Times New Roman', serif; /* Set font style */
                    text-align: center; /* Center align text */
                }
            </style>
            <p class="custom-text">Your Income is Less Than 50K</p>
        """, unsafe_allow_html=True)