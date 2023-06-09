import json
import os
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine, text


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

'''

correct code, but it doesn't work on my computer and gives an AttributeError: 'OptionEngine' object has no attribute 'execute'
engine = create_engine('sqlite:///{}'.format('data/DisasterResponse.db'))
df = pd.read_sql('DisasterResponse', engine)

sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
Background on this error at: https://sqlalche.me/e/20/e3q8
'''


engine = create_engine('sqlite:///{}'.format('/Users/bepresent/Desktop/disaster/data/DisasterResponse.db'))
df = pd.DataFrame(engine.connect().execute(text("SELECT * FROM DisasterResponse")).fetchall())

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = list(df.columns[4:])
    category_boolean = df.iloc[:,4:].astype(bool).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    # Graph 1 - Distribution of Message Genres
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hovertext=genre_counts,
                    marker=dict(
                        colors=['#d0f0c0', '#f0c0d0', '#c0d0f0']
                    )
                )
            ],

            'layout': {
                'title': 'Distribution of Genres',
            }
        },

        # Graph 2 - Distribution of Categories
        {
            'data': [
                Bar(
                    x = category_names,
                    y = category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 26
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()