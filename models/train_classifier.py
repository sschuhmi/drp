import sys, re, nltk, pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# Further specific imports for tokenizer:
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Specific imports for Pipeline:
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
# Import of Estimators:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
# Specific import for split into training & test set, and GridSearchCV:
from sklearn.model_selection import train_test_split, GridSearchCV
# Import of classification report (if needed):
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ Loads stored message & category data from a SQLite database.
    
        Parameters:
        - database_filepath (string): The filepath of the database file.
        
        Returns:
        - X (pandas.DataFrame): Input data (i.e., disaster messages)
        - Y (pandas.DataFrame): Output date (i.e., classification according to the given disaster message categories)
        - category_names (list of strings): The names of the classification categories
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    # Take the message itself as the input variable, i.e. X:
    X = df['message']
    # ...and the 36 categories, i.e. all columns except the 4 mentioned here, 
    # as output classification results, i.e. Y:
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])     
    # Load category names from classification output Y:
    category_names = Y.columns
    # Return everything to caller:
    return X, Y, category_names

def tokenize(text):
    """ Function to tokenize/clean the text messages.
    
        Proceeds the following steps:
        1. Tokenize text in words
        2. Lemmatize tokenized words
        3. Normalize case & strip white spaces (if any included)
        
        Parameters:
        - text (string): The text that has to be cleaned by this function.
        
        Returns:
        - cleaned_tokens (list of strings): The cleaned (i.e., tokenized, lemmatized, case-normalized and stripped) tokens
    """
    # Step 1: Tokenize text in words:
    tokens = word_tokenize(text)
    # Step 2: Lemmatize tokenized text:
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for tok in tokens:
        # Step 3: In combination with lemmatization, normalize case & strip white spaces (if any)
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
        
    return cleaned_tokens

def build_model():
    """ Function to build the model for prediction.
        This function creates a pipeline with several stages that includes 
        a CountVectorizer, a TfIdfTransformer and an AdaBoost base classifier 
        with optimized parameters that were previously determined 
        in a Jupyter Notebook using GridSearchCV.
       
       Parameters:
       - None
       
       Returns: 
       - model (Pipeline): The created pipeline used for training and prediction in the following.
    """
    # According to the preparational notebook, use Pipeline with CountVectorizer using 'tokenize' function, TfIdfTransformer and AdaBoost as base Classifier with the parameters determined by GridSearchCV:
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=None))) 
    ])
  
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """ Function the evaluate the previouls established model.
        Takes the model, the test data (input X_test, output Y_test), 
        the determined category names and predicts values. 
        Finally, prints out a classification report per category.
        
        Parameters:
        - model (pipeline): The previouls built model using a pipeline
        - X_test (list of messages): Input test data
        - Y_test (list of prediction values per category, i.e. 0's and 1's): Output test data 
            for classification per category
        - category_names (list of strings): The names of the classification categories
        
        Returns:
        - Nothing
    """
    # Predict values for model:
    Y_pred = model.predict(X_test)
    
    # Print out classification report per category:
    i = 0
    for category in category_names:
        print((i+1),'.',category,':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], target_names=category_names))
        i += 1
    
    return


def save_model(model, model_filepath):
    """ Function to save the model into a Pickle file
    
        Parameters:
        - model (pipeline): The model to be saved
        - model_filepath (string): The relative filepath of the Pickle file to be stored.
        
        Returns:
        - Nothing
    """
    # Save classifier with pickle:
    with open('classifier.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return


def main():
    """ Main function of process data. 
        Includes the workflow for load, cleaning and storing the data to a database.
        
        Parameters:
        No input or output parameters included, but has to be called with the following 
        arguments in the calling string:
        - filepath of the SQLite database where the loaded and cleaned data was stored.
        - filepath of the model stored in a Pickle file
        
        Returns:
        - Nothing
    """
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