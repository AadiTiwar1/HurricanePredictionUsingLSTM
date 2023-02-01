import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from PIL import Image


LATITUDE_CSV_MAIN = pd.read_csv('static/dataset/latitude.csv', sep=',', index_col=False)
LONGITUDE_CSV_MAIN = pd.read_csv('static/dataset/longitude.csv', sep=',', index_col=False)
MAX_SUSTAINED_WIND_CSV_MAIN = pd.read_csv('static/dataset/windspeed.csv', sep=',', index_col=False)

LATITUDE_CSV = LATITUDE_CSV_MAIN.loc[1107:].copy()
LONGITUDE_CSV = LONGITUDE_CSV_MAIN.loc[1107:].copy()
MAX_SUSTAINED_WIND_CSV = MAX_SUSTAINED_WIND_CSV_MAIN.loc[1107:].copy()

START_DATE = datetime.datetime.strptime(LATITUDE_CSV.loc[LATITUDE_CSV['Date'] != np.nan]['Date'].values[0], '%Y-%m-%d')
END_DATE = datetime.datetime.strptime(LATITUDE_CSV['Date'].iloc[-1], '%Y-%m-%d')

@st.cache

def convert_df(df: pd.DataFrame):
    return df.to_csv().encode('utf-8')
backgroundColor = '#273346'

# Explanation of the app
st.markdown("<h1 style='text-align: center;'>Hurricane Prediction</h1>", unsafe_allow_html=True)
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

st.header("");



st.markdown(
    """
    
        Predicting the windspeed and latitude/longitude of a hurricane from a certain inputed date
        
        ** Due to the fact that our data set ended in 2015, our model makes predictions up to 2017 
        (Techniqually in the future). Our model can also make prediction until 2035 but we decreased 
        it to lower training time for easier testing**

        In the following code, we train an LSTM model to predict future hurricanes, 
        based on a past dataset. The learning rate was
        decided after some experimentation, where we chose the learning rate 
        that gave accurate results. The number of epochs we use is 100, by which 
        the model converges. After that, we visualize three different graphs: the 
        longitude, latitude , and wind speed prediction
            """
)

st.header("");


st.download_button(
    label="Download Historical Data (.csv)",
    data=convert_df(pd.read_csv('static/dataset/atlantic.csv')),
    file_name='atlantic.csv',
    mime='text/csv',
)

st.markdown("![Foo](https://media.discordapp.net/attachments/1047322957212561478/1069875343906832414/e8X0fsrAi7EAAAAASUVORK5CYII.png)")
st.markdown("![Foo](https://media.discordapp.net/attachments/1047322957212561478/1069875278714773536/image.png)")

#* Section 2 is where the user can make predictions
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

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


#* Section 3 is where the user can see the prediction
st.subheader("Predicted Hurricane Information")
st.markdown(f"""
    Predicting for 
    <b style="color: teal;">{input_date}</b> 
""", unsafe_allow_html = True)

if st.session_state.pred_windspeed is not None:
    category = 0
    # Categorize the hurricane based on its windspeed
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
{"No Hurricane" if category == 0 else f"Category {category} hurricane"} ({st.session_state.pred_windspeed} knots or {st.session_state.pred_windspeed * 1.151} mph)
Latitude: {st.session_state.pred_lat}
Longitude: {st.session_state.pred_long}
    """, language="python")

st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        
        

# **** FOR REFERENCE ****
# ? GH LINK: https://github.com/cloudera/CML_AMP_Intelligent_Writing_Assistance
# ? Streamlit Docs: https://docs.streamlit.io/en/stable/
