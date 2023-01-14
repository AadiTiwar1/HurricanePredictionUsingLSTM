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
    accuracy = (1 - (np.sum(np.absolute(model.df_out["windspeed"] - model.df_out["Model Forecast"])) / np.sum(model.df_out["windspeed"]))) * 100

    st.code(f'Model Accuracy: {accuracy}%', language='text')

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

    # ********************************************* #
    #? maybe try using matplotlib to plot the graph??
    # fig = plt.figure(figsize=(12, 7))
    # plt.plot(range(len(model.df_out)), model.df_out["windspeed"], label = "Real")
    # plt.plot(range(len(model.df_out)), model.df_out["Model Forecast"], label = "LSTM Prediction")
    # plt.ylabel('Wind Speed')
    # plt.xlabel('Days')
    # plt.vlines(model.size, ymin=0, ymax=100, label = "Test set start", linestyles = "dashed")
    # plt.legend()
    # st.pyplot(fig)
    # ********************************************* #

    # TODO: features = ['central_pressure', 'latitude', 'longitude', 'max_sustained_wind'] => create inputs and a form to ask the user to input values


# Section 3 is where the user can make predictions

# **** FOR REFERENCE ****
# ? GH LINK: https://github.com/cloudera/CML_AMP_Intelligent_Writing_Assistance
# ? Streamlit Docs: https://docs.streamlit.io/en/stable/