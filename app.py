#Gender ==>>> 1 female 0Male
#Churn ==> 1 Yes 0  No
#Scaler is exported as scaler.pkl
#Model is exported aas model.pkl
#Order of X==>'Age', 'Gender', 'Tenure', 'MonthlyCharges'
import streamlit as st
import joblib
import numpy as np
st.title("Churn Prediction App")
st.write("Please enter a value for predict churn")
st.divider()
age=st.number_input("Enter Age",min_value=10,max_value=100,value=30)

tenure=st.number_input("Enter Tenure",min_value=0,max_value=130,value=10)
monthlycharges=st.number_input("Enter Monthly Charges",min_value=30,max_value=150)
gender=st.selectbox("Enter  the gender",["Male","Female"])
st.divider()
scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")
predict_button=st.button("Predict Churn")
if predict_button:
    gender_selected=1 if gender=="Female" else 0
    X=[age,gender_selected,tenure,monthlycharges]
    X1=np.array(X)
    X_array=scaler.transform([X1])
    prediction=model.predict(X_array)[0]
    predicted="Yes" if prediction==1 else "No"
    st.balloons()
    st.write(f"The predicted value is : {predicted}")
else:
    st.write("Please Enter a value and use Predict Button")