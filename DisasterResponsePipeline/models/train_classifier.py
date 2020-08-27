import sys
import pandas as pd
import numpy as np
import pickle
import re
import time
import warnings

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'wordnet', 'stopwords'])
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    '''
    Load data from SQL Database
    
    Arguments:
        - database_filepath: SQL database file
    
    Returns:
        - X pandas dataframe: Features
        - Y pandas dataframe: Targets
        - category_names list: category labels 
    '''
    
    # connect to database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read table into df dataframe
    query = "SELECT * FROM InsertTableName;"
    df = pd.read_sql(query, con=engine)
    
    # split dataframe into X and Y
    X = df['message']
    Y = df.iloc[:,4:]
    
    # get category column names
    cat_names = list(Y.columns)
    
    return X, Y, cat_names

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
    lemmed = [WordNetLemmatizer().lemmatize(w.strip()) for w in words]
    
    # Lemmatization 2: Reduce verbs to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w.strip(), pos='v') for w in lemmed]
    
    return lemmed


def build_model():
    '''
    Build Model with Grid Search
    
    Returns:
        - Tuned Model after grid search
    '''

    mclf = MultiOutputClassifier(OneVsRestClassifier(LinearSVC()))

    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', mclf)
                        ])
    
    # hyper-parameter grid
    parameters = {
              'vect__max_df': [0.75, 1.0],
              #'vect__min_df': (0.5, 0.75, 1.0),
              'tfidf__smooth_idf': [True],
              'clf__estimator__estimator__max_iter': [500, 750, 1000]
                 }

    # create model
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model performance evaluation on test data
    
    Arguments:
        - model: trained model
        - X_test: Test features
        - Y_test: Test targets
        - category_names: Target labels

    Output:
        Classification metrics for each category and overall accuracy
    '''

    # predict
    y_pred = model.predict(X_test)

    lbls = list(range(36))

    # print classification report
    print("{: >23}\t{: <}\t{: <}\t{: <}\t\t{: <}".format('Category', 'Accuracy', 'Precision', 'Recall', 'F1 Score'))

    for i in range(len(category_names)):

        print("{: >23}\t{:>.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
                                                        category_names[i]+':',
                                                        accuracy_score(Y_test.values[:, i], y_pred[:, i]),
                                                        precision_score(Y_test.values[:, i], y_pred[:, i], average='weighted'),
                                                        recall_score(Y_test.values[:, i], y_pred[:, i], average='weighted'),
                                                        f1_score(Y_test.values[:, i], y_pred[:, i], average='weighted')
                                                                     )
             )

    # print accuracy score
    print('\n{: >23}\t{:.6f}'.format('Overall Accuracy:', np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    '''
    Save model to a Python pickle file    
    Arguments:
        - model: Trained model
        - model_filepath: Filepath to save the model
    
    Output:
        - file as specified in model_filepath for model
    '''

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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