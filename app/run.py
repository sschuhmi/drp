import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
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
engine = create_engine('sqlite:///../data/drp.db')
df = pd.read_sql_table('messages', engine)

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
    
    # Descending categorical message counts for ALL messages (for 2nd visualization):
    cats =  df[df.columns[4:]] # select categories within DataFrame
    cat_counts = (cats.shape[0] * cats.mean()).sort_values(ascending=False)
    cat_names = list(cat_counts.index)
    
    # Descending categorical message counts only for NEWS messages (for 3nd visualization):
    cats_news =  df[df.genre == 'news'].drop('id', axis=1) # select NEWS genre within df, drop ID col
    cat_news_counts = (cats_news.shape[0] * cats_news.mean()).sort_values(ascending=False)
    cat_news_names = list(cat_news_counts.index)
    
    # Descending categorical message counts only for DIRECT messages (for 4th visualization):
    cats_dir =  df[df.genre == 'direct'].drop('id', axis=1) # select DIRECT genre within df, drop ID col
    cat_dir_counts = (cats_dir.shape[0] * cats_dir.mean()).sort_values(ascending=False)
    cat_dir_names = list(cat_dir_counts.index)
    
    # Descending categorical message counts only for SOCIAL messages (for 5th visualization):
    cats_soc =  df[df.genre == 'social'].drop('id', axis=1) # select SOCIAL genre within df, drop ID col
    cat_soc_counts = (cats_soc.shape[0] * cats_soc.mean()).sort_values(ascending=False)
    cat_soc_names = list(cat_soc_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Plotting of messages according to categories in descending order (2nd visualization)
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Message Category Distribution - ALL messages',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # ...filtered on NEWS (3rd visualization)
        {
            'data': [
                Bar(
                    x=cat_news_names,
                    y=cat_news_counts
                )
            ],
        
            'layout': {
                'title': 'Distribution of Message Categories - only NEWS messages',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }     
        },
        # ...filtered on DIRECT (4th visualization)
        {
            'data': [
                Bar(
                    x=cat_dir_names,
                    y=cat_dir_counts
                )
            ],
        
            'layout': {
                'title': 'Distribution of Message Categories - only DIRECT messages',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }     
        },
        # ...filtered on SOCIAL (5th visualization)
        {
            'data': [
                Bar(
                    x=cat_soc_names,
                    y=cat_soc_counts
                )
            ],
        
            'layout': {
                'title': 'Distribution of Message Categories - only SOCIAL messages',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
