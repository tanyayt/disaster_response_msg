import json
import plotly
import pandas as pd
import joblib
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_messages', engine)
print("-----database loaded-----")

# load model
model = joblib.load("../model/trained_model.pkl")
print("-----model loaded------")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    df.iloc[:,4:]=df.iloc[:,4:].astype(int)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #caegory counts
    category_counts=df.iloc[:,4:].sum().sort_values(ascending=False)
    
    #create word counts 
    word_counts=df['message'].apply(lambda x:len(re.findall(r"[\w]+",x.lower())))
    
      
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of Categories in Disaster Response Message Dataset',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "" 
                }
            }
        },
        
        #second graph test
        {
            'data': [
                {
                    "type":"histogram",
                    "x": word_counts
     
                }
            ],

            'layout': {
                'title': 'Distribution of Word Counts in Disaster Response Message Dataset',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Word Counts in Each Message",
                    'range':[0,100]
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
    app.run(host='127.0.0.1', port=800, debug=True)


if __name__ == '__main__':
    main()