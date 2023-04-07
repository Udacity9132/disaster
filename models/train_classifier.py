import sys
import pandas as pd
import re
from sqlalchemy import create_engine, text

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle




def load_data(database_filepath):
    '''
    Input: Loads cleaned data from the SQLite Database
    Output: 
    X - input variable: messages 
    Y - output variable: categories of the messages 
    category_names - category name for Y
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.DataFrame(engine.connect().execute(text("SELECT * FROM DisasterResponse")).fetchall())

    # Define features and targets
    X = df.message
    Y = df.loc[:, 'related':'direct_report']

    # Define category names for targets
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize message data

    Input: message text data as string
    Output: Array of tokenized messages as dataframe
    '''
    # transform text to lowercase and remove punuctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize

    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return words
    


def build_model():
    '''
    SciKit ML Pipeline which procceses the text messages and applies a classifier
    
    The pipeline consists of three main steps:
    1. CountVectorizer: Convert text into a matrix of token counts
    2. TfidfTransformer: Transform a count matrix to a normalized tf-idf (Term frequency-Inverse document frequency) representation
    3. MultiOutputClassifier: Create a multi-label classification model using scikit-learn's RandomForestClassifier
    '''
    moc = MultiOutputClassifier(RandomForestClassifier(n_estimators=50))

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
    ])

    # specify the parameters for grid search
    parameters = {'clf__estimator__max_depth': [10, 50],
                  'clf__estimator__min_samples_leaf':[2, 5]}
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, verbose=2)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the performance of a machine learning model using classification_report
    Input:
    model - our trained scitkit-learn model
    X_test - test set features
    Y_test - test set categories
    category_names - category name for Y

    Output: none - classification_report with precision, recall, and F1-score
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'-----------------------{i, col}----------------------------------')
        print(classification_report(list(Y_test.values[:, i]), list(y_pred[:, i])))



def save_model(model, model_filepath):
    '''
    Input: stored classification model + filepath to pickle file
    Output: none - saved pickle file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    '''
    Train and evaluate a machine learning model

    INPUT:
    None - The function expects command line arguments to specify the filenames for the input data and output model

    OUTPUT:
    None - The function saves the trained model to a file and prints evaluation metrics to the console
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()