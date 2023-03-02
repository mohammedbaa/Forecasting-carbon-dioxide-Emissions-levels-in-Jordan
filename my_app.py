# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:33:10 2023

@author: mohammed
"""
import streamlit as st
import numpy as np 
import pandas as pd 
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima_model import ARIMAResults
import warnings
import pickle
warnings.filterwarnings('ignore')



def load_model():
    model = pickle.load(open('C:/Users/mohammed/Desktop/Forecasting-carbon-dioxide-Emissions-levels-in-Jordan-main/Forecast_arima.pkl', 'rb'))
    return model

#@st.cache()  # Cache the data
def load_data():
    df = pd.read_csv("C:/Users/mohammed/Desktop/Forecasting-carbon-dioxide-Emissions-levels-in-Jordan-main/updatedco2.csv")
    return df

def Arima_pred(df, f):
    df.index = pd.to_datetime(df.index)
    last_date = df.index[-1]
    offsets = [DateOffset(years=x) for x in range(1, f+1)]
    future_dates = [offset.apply(last_date) for offset in offsets]
    new_data = pd.DataFrame(index=future_dates)
    model = load_model()
    predictions = model.predict(start=len(df), end=len(df)+len(new_data)-1, exog=new_data)
    return predictions



if __name__ == '__main__':
    # Load data
    data = load_data()

    # Set up Streamlit app
    st.title("ARIMA Model Predictions")

    # Display data
    st.write("Data:")
    st.write(data)

    # Get user input for number of years to predict
    f = st.slider("Select the number of years to predict", 1, 10)

    # Make predictions and display results
    if st.button("Make Predictions"):
        if f <= 10:
            predictions = Arima_pred(data, f)
            st.write(f"Predictions for the next {f} years:")
            st.write(predictions)
        else:
            st.write("Invalid input. Please enter a value between 1 and 10.")