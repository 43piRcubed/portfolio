import json
import plotly
import plotly.graph_objs as gobj
import pandas as pd
import numpy as np
import operator
import re
import math

from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib  -  DEPRECATED
import joblib
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords'])
app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize text data
    
    Arguments:
        - text str: Messages as text data
    
    Returns:
        - words list: Processed text after normalizing, 
                                           tokenizing,
                                           removing stopwords,
                                           lemmatizing: words & verbs
    '''
    # convert all text to lower case & removing punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize words
    words = word_tokenize(text)
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization 1: Reduce words to their root form
    clean_tok = [WordNetLemmatizer().lemmatize(w.strip()) for w in words]
    
    # Lemmatization 2: Reduce verbs to their root form
    clean_tok = [WordNetLemmatizer().lemmatize(w.strip(), pos='v') for w in clean_tok]
    
    return clean_tok


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # Ordinates and variables for 1st Graph
    # data to show the distribution of all message by genre
    genre_counts = df.groupby('genre').count()['message'] # Y: counts of genre
    genre_names = list(genre_counts.index)                # X: genre
    
    # Ordinates and variables for 2nd Graph
    # data to show frequency of the message categories
    # in proportion to the total number of messages
    cat_prop = 100 * df[df.columns[4:]].sum()/len(df)        # Y: category proportion based on total count %
    cat_prop = cat_prop.sort_values(ascending = False)       # Sort in descending order
    cats = list(cat_prop.index)                              # X: category names
    
    # Ordinates and variables for 3rd Graph
    # this is to show proportion of message categories
    # broken down by genre
    cat_prop_genre = df[df.columns[3:]].groupby('genre').sum() # getting category counts grouped by genre
    cat_prop_genre.loc['total'] = cat_prop_genre.sum()         # Adding total row for sorting
    cat_prop_genre = cat_prop_genre.sort_values(by = 'total', ascending = False, axis = 1) # sort on total row descending
    cat_prop_genre = cat_prop_genre.reset_index(level = 0)     # Y: dataframe containing category counts per genre
    cats_g3 = list(cat_prop_genre.columns)[1:]                 # X: category names
    
    # create visuals
    
    graphs = [
             {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                   )
                    ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
                      }
             },
             {
            'data': [
                Bar(
                    x=cats,
                    y=cat_prop
                   )
                    ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion [%]",
                    'automargin':True
                         },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35,
                    'automargin':True
                         }
                      }
             },
             {
            'data': [gobj.Bar(
                x = cats_g3,
                y = 100 * cat_prop_genre[cat_prop_genre.genre == g][cat_prop_genre.columns[1:]].iloc[0] / len(df),
                name = g
                             )
                     for g in genre_names
                    ],

            'layout': {
                'title': 'Proportion of Messages by Category and Genre',
                'yaxis': {
                    'title': "Proportion [%]",
                    'automargin':True,
                         },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -35,
                    'automargin':True
                         },
                'barmode': 'stack'
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