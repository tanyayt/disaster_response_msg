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
    Returns: a pipeline for model training
    '''
       
    # Build a machine learning pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    #omit tuning to improve performance
           
    return pipeline
    

def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_test_predicted=model.predict(X_test)
    
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_test_predicted[:, i]))
        print(accuracy_score(Y_test.iloc[:, i].values, Y_test_predicted[:, i]))
        print("-"*30)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
        
        print('Loading data...\n    DATABASE: {}'.format(my_database_filepath))
        X, Y, category_names = load_data(my_database_filepath)
                
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model..this may take about 20 minutes')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(my_model_filepath))
        save_model(model, my_model_filepath)
        
        print('Trained model saved!')

if __name__ == '__main__':
    main()