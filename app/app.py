import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

MODEL_TRAINED_STATE = os.path.exists('src/hurricane_model.pt')

# add images of the prediction graph and the loss function graphs (put them in static/images folder)
# connect to model in src/model.py and run the predict function when the user submits their values to predict on
# add fields in the streamlit app to ask the user to submit values

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

# Section 2 shows graphs of  the prediction graph, model accuracy
def temp_predict_fn(date, latitude, longitude, central_pressure, max_sustained_wind):
    print(f"Predicting for {date} at {latitude}, {longitude} with {central_pressure} mb and {max_sustained_wind} kt...")

def set_model_loaded():
    st.session_state.loaded = True

load_model_btn = st.button("Load Model" if MODEL_TRAINED_STATE else "Train Model", on_click=set_model_loaded, disabled=st.session_state.loaded if 'loaded' in st.session_state else False)
if 'loaded' not in st.session_state:
    st.session_state.loaded = False

if load_model_btn or st.session_state.loaded:
    with st.spinner(('Loading Model...' if MODEL_TRAINED_STATE is True else 'Training Model...')):
        import src.model as model
        # calculate the accuracy of the model
        st.code(f'Model Accuracy: {model.accuracy}%', language='ruby')

        # create the graphs of the data and predicted data
        chart_data_windspeed = pd.DataFrame(
            model.df_out,
            columns = ["next_windspeed"]
        )

        chart_data_predict = pd.DataFrame(
            model.df_out,
            columns = ["Model Forecast"]
        )

        st.header("Historical Data")
        show_historical_data = st.checkbox("Show/Hide Historical Data")
        if show_historical_data:
            st.error("Uncheck the checkbox above if you don't need to see the historical data anymore as it may slow down the app.", icon="⚠")
            st.line_chart(chart_data_windspeed)

        st.header("Predicted Data")
        show_predicted_data = st.checkbox("Show/Hide Predicted Data")
        if show_predicted_data:
            st.error("Uncheck the checkbox above if you don't need to see the predicted data anymore as it may slow down the app.", icon="⚠")
            st.line_chart(chart_data_predict)
        # TODO: create a vertical line to show where the test set starts

        # ************************************************************************************************** #
        # ? maybe try using matplotlib to plot the graph to show the vertical line where the test set starts ?
        # fig = plt.figure(figsize=(12, 7))
        # plt.plot(range(len(model.df_out)), model.df_out["windspeed"], label = "Real")
        # plt.plot(range(len(model.df_out)), model.df_out["Model Forecast"], label = "LSTM Prediction")
        # plt.ylabel('Wind Speed')
        # plt.xlabel('Days')
        # plt.vlines(model.size, ymin=0, ymax=100, label = "Test set start", linestyles = "dashed")
        # plt.legend()
        # st.pyplot(fig)
        # ************************************************************************************************** #

    #* Section 3 is where the user can make predictions
    # DATE
    st.header("Create your own prediction")
    st.markdown("""
        #### Date
        The date of the hurricane. Format: YYYYMMDD
    """, unsafe_allow_html = True)
    current_date = int(datetime.datetime.now().strftime("%Y%m%d"))
    input_date = st.number_input('Enter a value', min_value=0, value=current_date, step=1, help="Date of the hurricane. Format: YYYYMMDD", format="%d", key="date")

    # LATITUDE
    st.markdown("""
        #### Latitude
        The latitude of the hurricane. Format: -90.0 (S) to +90.0 (N)
    """, unsafe_allow_html = True)
    input_lat = st.slider('Select a value', min_value=-90.0, max_value=90.0, value=0.0, step=0.1, help="The latitude of the hurricane. Format: -90 (S) to +90 (N)", format="%.1f", key="lat")

    # LONGITUDE
    st.markdown("""
        #### Longitude
        The longitude of the hurricane. Format: -180.0 (W) to +180.0 (E)
    """, unsafe_allow_html = True)
    input_long = st.slider('Select a value', min_value=-180.0, max_value=180.0, value=0.0, step=0.1, help="The longitude of the hurricane. Format: -180 (W) to +180 (E)", format="%.1f", key="long")

    # CENTRAL PRESSURE
    st.markdown("""
        #### Central Pressure
        At sea level, standard air pressure in millibars is
        <b style="font-weight: bolder; color: teal;">1013.2</b>
        (weather.gov).

        The lower the pressure, the more severe the hurricane. Normally, during a hurricane, 
        the pressure is between 900 and 1000.

        Category 1 - Minimal: > 980 millibars <br/>
        Category 2 - Moderate: 965 - 979 millibars <br/>
        Category 3 - Extensive: 945 - 964 millibars <br/>
        Category 4 - Extreme: 920 - 944 millibars <br/>
        Category 5 - Catastrophic: < 920 millibars <br/>
    """, unsafe_allow_html = True)
    input_central_pressure = st.slider('Select a value in millibars', min_value=0.0, max_value=1500.0, value=1013.2, step=0.1, help="Central Pressure of the hurricane. Units in millibars.", key="central_pressure", format="%.1f")

    # CENTRAL PRESSURE
    st.markdown("""
        #### Maximum Sustained Wind Speed
        The maximum sustained wind speed of the hurricane. Units in knots.

        Category 1 - Minimal: 64 - 83 kt <br/>
        Category 2 - Moderate: 83 - 96 kt <br/>
        Category 3 - Extensive: 96 - 113 kt <br/>
        Category 4 - Extreme: 113 - 135 kt <br/>
        Category 5 - Catastrophic: > 135 kt <br/>
    """, unsafe_allow_html = True)
    input_max_sustained_wind = st.slider('Select a value in knots', min_value=0.0, max_value=200.0, value=0.0, step=0.1, help="The maximum sustained wind speed of the hurricane. Units in knots.", key="max_sustained_wind", format="%.1f")

    # SUBMIT BUTTON
    st.button("Predict", key="submit", on_click=temp_predict_fn, args=(input_date, input_lat, input_long, input_central_pressure, input_max_sustained_wind))



# **** FOR REFERENCE ****
# ? GH LINK: https://github.com/cloudera/CML_AMP_Intelligent_Writing_Assistance
# ? Streamlit Docs: https://docs.streamlit.io/en/stable/