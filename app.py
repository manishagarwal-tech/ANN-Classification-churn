import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

#load the model
model = tf.keras.models.load_model('model.h5')

# load all encoders and scaler
with open('onehotencoder.pkl', 'rb') as f:
    onehotencoder = pickle.load(f)

with open('labelencoder.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



# streamlit app
st.title("Custom Churn Predictor")
#User Input and create a form
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', labelencoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure' : [tenure],
        'Balance' : [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
})

#convert encoder  for geography and gender
input_data['Gender'] = labelencoder.transform(input_data['Gender'])
geo_encoder = onehotencoder.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder, columns=onehotencoder.get_feature_names_out(['Geography']))

#combine with current input data
input_df = pd.concat([input_data.drop('Geography', axis=1), geo_encoder_df], axis=1)

# scale the data:
input_scaled_data = scaler.transform(input_df)
prediction = model.predict(input_scaled_data)
prediction_prob = prediction[0][0]
# print the prod score
st.write(f"Churn Probability: {prediction_prob:.2f}")
if prediction_prob > 0.5:
    st.write("The custom is likely to churn")
else:
    st.write("The churn will not be likely to churn")

