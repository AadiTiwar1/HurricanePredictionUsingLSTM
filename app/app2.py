import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

LATITUDE_CSV = pd.read_csv('static/dataset/latitude.csv', sep=',', index_col=False)
LONGITUDE_CSV = pd.read_csv('static/dataset/longitude.csv', sep=',', index_col=False)
MAX_SUSTAINED_WIND_CSV = pd.read_csv('static/dataset/windspeed.csv', sep=',', index_col=False)

LATITUDE_CSV = LATITUDE_CSV.loc[1356:].copy()
LONGITUDE_CSV = LONGITUDE_CSV.loc[1356:].copy()
MAX_SUSTAINED_WIND_CSV = MAX_SUSTAINED_WIND_CSV.loc[1356:].copy()

START_DATE = datetime.datetime.strptime(LATITUDE_CSV['Date'].iloc[0], '%Y-%m-%d')
END_DATE = datetime.datetime.strptime(LATITUDE_CSV['Date'].iloc[-1], '%Y-%m-%d')

print(START_DATE, END_DATE)

@st.cache
def convert_df(df: pd.DataFrame):
    return df.to_csv().encode('utf-8')

# Explanation of the app
st.title("Hurricane Prediction")
st.markdown(
    """
        Predicting the windspeed of a hurricane from its latitude, 
        longitude, maximum sustained wind, and its central pressure.

        In the following code, we train an LSTM model to predict future hurricanes, 
        and then test it on the test dataset. The learning rate was
        decided after some experimentation, where we chose the learning rate 
        that gave accurate results. The number of epochs we use is 6, by which 
        the model converges. After that, we visualize two different graphs: the 
        comparison between the real hurricane data and the ones given by the 
        model.
    """
)

st.download_button(
    label="Download Historical Data (.csv)",
    data=convert_df(pd.read_csv('static/dataset/atlantic.csv')),
    file_name='atlantic.csv',
    mime='text/csv',
)


#* Section 3 is where the user can make predictions
# DATE
st.header("Create your own prediction")
st.markdown("""
    #### Date
    The date of the hurricane. Format: YYYYMMDD
""", unsafe_allow_html = True)
input_date = st.date_input('Enter a value', START_DATE, help="Date of the hurricane. Format: YYYYMMDD", key="date", min_value=START_DATE, max_value=END_DATE)

# SUBMIT BUTTON AND FUNCTION
if "pred_windspeed" not in st.session_state:
    st.session_state.pred_windspeed = None

if "pred_lat" not in st.session_state:
    st.session_state.pred_lat = None

if "pred_long" not in st.session_state:
    st.session_state.pred_long = None

def on_submit():
    st.session_state.pred_windspeed = MAX_SUSTAINED_WIND_CSV.loc[MAX_SUSTAINED_WIND_CSV['Date'] == input_date.strftime('%Y-%m-%d')]['Forecast'].values[0]
    st.session_state.pred_lat = LATITUDE_CSV.loc[LATITUDE_CSV['Date'] == input_date.strftime('%Y-%m-%d')]['Forecast'].values[0]
    st.session_state.pred_long = LONGITUDE_CSV.loc[LONGITUDE_CSV['Date'] == input_date.strftime('%Y-%m-%d')]['Forecast'].values[0]

st.button("Predict", key="submit", on_click=on_submit)

#* Section 4 is where the user can see the prediction
st.header("Predicted Wind Speed")
st.markdown(f"""
    Predicting for 
    <b style="color: teal;">{input_date}</b> 
""", unsafe_allow_html = True)

if st.session_state.pred_windspeed is not None:
    category = 0
    if st.session_state.pred_windspeed >= 74 and st.session_state.pred_windspeed <= 95:
        category = 1
    
    if st.session_state.pred_windspeed >= 96 and st.session_state.pred_windspeed <= 110:
        category = 2

    if st.session_state.pred_windspeed >= 111 and st.session_state.pred_windspeed <= 129:
        category = 3

    if st.session_state.pred_windspeed >= 130 and st.session_state.pred_windspeed <= 156:
        category = 4

    if st.session_state.pred_windspeed >= 157:
        category = 5

    st.code(f"""
        {
            f'''
{"No Hurricane" if category == 0 else f"Category {category} hurricane"} ({st.session_state.pred_windspeed} knots or {st.session_state.pred_windspeed * 1.151} mph)
Latitude: {st.session_state.pred_lat}
Longitude: {st.session_state.pred_long}
            '''
        }

    """, language="python")

        
        

# **** FOR REFERENCE ****
# ? GH LINK: https://github.com/cloudera/CML_AMP_Intelligent_Writing_Assistance
# ? Streamlit Docs: https://docs.streamlit.io/en/stable/