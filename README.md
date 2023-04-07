# Disaster Response Pipeline Project

# Summary of Project

This project applies data engineering skills to analyze disaster data and build a model for an API that classifies disaster messages. Using a machine learning pipeline, the project categorizes real messages from disaster events and sends them to the appropriate relief agency. The project includes a web app that allows an emergency worker to input a message and receive classification results in multiple categories, along with visualizations of the data. The project showcases the ability to create basic data pipelines and write clean, organized code.

The project is segmented into three parts:

1. Developing an ETL pipeline to extract data, clean it, and store it in a SQLite database
2. Constructing a machine learning pipeline to train our model
3. Executing a web application to display our model's outcomes

# Data

The project's data originates from Figure Eight's Multilingual Disaster Response Messages, comprising 30,000 messages derived from various disaster events worldwide. These events range from the earthquake in Haiti and Chile in 2010 to floods in Pakistan in 2010 and super-storm Sandy in the USA in 2012. The dataset comprises news articles covering numerous disasters over several years. It contains 36 categories related to disaster response, with all sensitive information removed.

The data consists of two CSV files:

- disaster_messages.csv: messages data 
- disaster_categories.csv: disaster categories

# Explanation of the files

process_data.py (ETL Pipeline):

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

train_classifier.py (ML Pipeline):
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

run.py (Flask App): 
is responsible for executing the Flask application that categorizes messages according to the model and displays data visualisations

# Instructions:

### Dependencies

- Machine Learning Libraries and Processes: Pandas, Numpy, Scipy, Scikit-Learn, NLTK, Pickle
- SQLite Database: SQLalchemy
- Web App and Visualisations: Flask, Plotly 
- Python 3+ is required

If you haven't installed this yet on your machine, you might have to run the following first:

$ pip install SQLAlchemy
$ pip install nltk
$ pip install joblib

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements

Dataset provided by Appen (formerally Figure-Eight)
Code template provided by Udacity 
