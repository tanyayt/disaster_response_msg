# import packages

import sys
import pandas as pd
import sqlite3
import re
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Deafault database filepath and model file path 
# Edit default paths as needed
my_database_filepath="..\data\disaster_response.db"
my_model_filepath="trained_model.pkl"

def load_data(database_filepath):
    '''
    Load data from database
    Args: database's filepath
    Returns: feature data frame X, Y, and category_names(list)
    '''
    connection=sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * from disaster_messages",connection)
    X = df['message']
    Y = df.iloc[:,4:].astype(int)
    category_names= Y.columns
    return X,Y, category_names

def tokenize(text):
    '''
    Args: text string
    Returns: a list of clean tokens
    '''
    #Replace urls with 'urlplaceholder' so urls are not tokenized
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #find all urls and store in all_urls
    all_urls = re.findall(url_regex,text)
    
    for url in all_urls:
        text=text.replace(url,"urlplaceholder")
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #crate a lemmatizer object 
    lemmatizer = WordNetLemmatizer()
    
    #clean tokens with lemmatizer
    clean_tokens =[]
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    '''
    Args: None
    Returns: a trained model
    '''
       
    # Build a machine learning pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    #set pramameters for tuning
    parameters = {'vect__ngram_range': ((1,1),(1, 2))
            , 'vect__max_df': (0.5, 0.75, 1.0)
            , 'vect__max_features':(None,50,100,200)
            , 'tfidf__use_idf': (True, False)
            , 'clf__estimator__n_estimators': [50, 100, 200]
            , 'clf__estimator__min_samples_split': [2, 3, 4]
        }

    # Optimize model using GridSearch 
    model = GridSearchCV(pipeline,param_grid=parameters)
        
    return model
    
    

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pickle.dump(cv, open(model_filepath, 'wb'))
    


def main():
    
    
        print('Loading data...\n    DATABASE: {}'.format(my_database_filepath))
        X, Y, category_names = load_data(my_database_filepath)
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(my_model_filepath))
        save_model(model, my_model_filepath)
        print('Trained model saved!')

if __name__ == '__main__':
    main()