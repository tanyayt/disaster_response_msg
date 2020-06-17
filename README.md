# Disaster Response Pipeline Project

## Motivation

The purpose of this project is to build a web app to classify disaster response messages and help emergency response team to quickly assess a new message and make decisions on what kind of help is needed. 

## Web App Screenshots

<img src="https://github.com/tanyayt/udacity_data_scientist/blob/master/disaster_response_msg/image/page_top.PNG?raw=true"> 

<img src="https://github.com/tanyayt/udacity_data_scientist/blob/master/disaster_response_msg/image/dataset_view1.PNG?raw=true">

<img src="https://github.com/tanyayt/udacity_data_scientist/blob/master/disaster_response_msg/image/dataset_view2.PNG?raw=true">



## Installation

*   Download the entire folder in your local drive

*   A database file with clean data has already been created and saved as `\data\disaster_response.db`

    *   If needed, you can re-run the ETL pipleline,using raw data. Make sure you set the working directory to the data folder

        ```
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
        ```

        

*   Train and save the model

    *   Set the working directory to the `\model` folder
    *   Run `python train_classifier.py../data/DisasterResponse.db classifier.py`
    *   This process can take about 4 hours and the trained model is saved in `\model` folder as `classifier.pkl` (~918M)
    *   (Optional) compress 'classifier.pkl' by running `compress_pickle.py classifier.pkl clasifier.pkl`

*   Build the web app
    *   Set the working directory to `\app` folder
    *   Run `python run.py`
    *   Once down, view and try the web app on your browser (http:\\\127.0.0.1:800)


# Files

*   [`process_data.py`](): this files contains the ETL pipeline

*   `\image` folder: contains screenshot images

*   `\model\train_classifier.py`the machine learning pipeline used to fit and evaluate model. The model is saved in a pickle file (not in this directory but you can generate in your local machine)

*   `\data\process_data.py` stores the ETL pipeline

*   `\data\disaster_messages.csv` and `\data\disaster_categories.csv` are raw response message data for training models

*   `\app` folder: contains files for building web apps 

*   `\app\templates\*.html`: HTML templates for web app

*   `\app`\run.py` starts the python server, loads data model, and build the web app



## References 

*   [Compressed Pickle](https://medium.com/better-programming/load-fast-load-big-with-compressed-pickles-5f311584507e)
*   

# Acknowledgement

This project is part of Udacity's Data Scientist Nanodegree program. Initial code templates are provided by [Udacity](www.udacity.com) and the disaster response data is provided by [Figure8](https://www.figure-eight.com/). 

