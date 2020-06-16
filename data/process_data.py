'''
process csv data and save it in a database
Input: messages_filepath, categories_filepath, database_filepath
Output:None

'''

# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# provide/modify filenames 
messages_filepath ="messages.csv"
categories_filepath="categories.csv"
database_filepath="disaster_response.db"
database_tablename="disaster_messages"


def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files 
    Args:
        messages_filepath: a string that contains the filepath of messages
        categories_filepath: a string that contains the filepath of messages    
    Returns: df - a Pandas dataframe with raw data
    '''
    
    # read in csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge data on id
    df=pd.merge(messages,categories,on="id")
    
    return df
 
def clean_data(df):
    '''
    clean data by splitting and renaming category columns
    Args: a dataframe returned from load_data
    Returns: a dataframe with clean data
    '''
    #split category columns 
    categories = df["categories"].str.split(';', expand = True)
    
    #use the non-numeric portion as the column names
    row = categories.iloc[0]
    category_colnames =row.apply(lambda x:x[:-2]) 
    categories.columns = category_colnames
    
    # convert each column to numeric 
    for column in categories.columns:
           categories[column] = categories[column].apply(lambda x:x.split('-')[1])
   
    categories[column] = pd.to_numeric(categories[column])
    
    #drop original category columns from df 
    df=df.drop("categories",axis=1)
    
    #concatenate df with numeric categories
    df=pd.concat([df,categories],axis=1)
    
    #drop duplicates 
    df=df.drop_duplicates()
   
    return df


def save_data(df,database_filepath):
    '''
    Saves a database file from df
    Args: database with clean data, database filename; the table name is database_tablename
    Returns: None;
    '''
    # load to database
    engine = create_engine("sqlite:///"+database_filepath)
    df.to_sql(database_tablename, engine, index=False)
   
def main():

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    

if __name__ == '__main__':
    main()