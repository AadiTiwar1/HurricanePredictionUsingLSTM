import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

MODEL_TRAINED_STATE = os.path.exists('C:/Users/Aarus/Documents/code/HurricanePredictionUsingLSTM/src/hurricane_model.pt')

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
        the model converges. After that, we visualize three different graphs: the 
        comparison between the real hurricane data and the ones given by the 
        model.
    """
)

# Section 2 shows graphs of  the prediction graph, model accuracy
with st.spinner(('Loading Model...' if MODEL_TRAINED_STATE is True else 'Training Model...')):
    import src.model as model

    # calculate the accuracy of the model
    st.code(f'Model Accuracy: {model.accuracy}%', language='text')

    # create the graphs
    chart_data_windspeed = pd.DataFrame(
        model.df_out,
        columns = ["windspeed"]
    )

    chart_data_predict = pd.DataFrame(
        model.df_out,
        columns = ["Model Forecast"]
    )

    st.header("Historical Data")
    st.line_chart(chart_data_windspeed)
    st.header("Predicted Data")
    st.line_chart(chart_data_predict)

    

    


