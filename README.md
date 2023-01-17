# Hurricane Windspeed Forecasting using Long Short Term Memory (LSTM) Recurrent Neural Networks (RNN)


Hurricanes remain as one of the deadliest natural disaster on the planet. Utilizing a Long Short Term Memory (LSTM) Recurrent Neural Network (RNN) to forecast the windspeeds of future hurricanes, our project aims to develop a unique, effective, and easy-to-use model that allows for various input parameters. To create predictions, you can use our interactive app contains sliders which allow you to precisely enter inputs such as dates to get an output in seconds for the windspeeds and location of the next big hurricane. Based on these predicted results, one can easily determine the severity and category of the predicted hurricane. This repo is inspired from the contents of this [repo](https://github.com/DikshantDulal/SoftServe_QLSTM) which uses LSTM for stock prediction. We utilized the LSTM aspect of the project to Hurricane data and developed different columns with various learning rates and epochs to maximize accuracy. 
# Instructions
This repo provides scripts and examples that can help you predict hurricanes on your own computer. The repo also provides a [Jupyter Notebook](https://github.com/AadiTiwar1/HurricanePredictionUsingLSTM/blob/main/src/HurricanePredictionDraft1%20(3).ipynb)) notebook which contains all of the code in one place with text descriptions for the easisest learning experience. [Note: We highly reccomend just downloading the Jupyter Notebook and running that instead of downloading the entire project and running the files!] For a more file-seperated based version, our Repo structure and brief description of content is provided below:

├── app/
├── scripts/
│   ├── download_data.py
│   ├── install_dependencies.py
│   ├── train_model.py
│   └── launch_app.py
├── src/    
├── static/dataset
├── .project-metadata.yaml
├── README.md
└── requirements.txt
The following scripts are recommended as a starting point:

Install dependencies
`pip3 install -r requirements.txt`
This repo depends on a few libraries (e.g. PyTorch) that need to be installed on the latest version.

