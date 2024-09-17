import sys
# Import Python and further libraries:
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load the message and categories data from the given filepaths 
        and merge them into a single Pandas DataFrame.
        
        Parameters: 
        - messages_filepath (string): Relative filepath of messages file (disaster_messages.csv)
        - categories_filepath (string): Relative filepath of categories file (disaster_categories.csv) 
        
        Returns: 
        - df (pandas.DataFrame): DataFrame with merged messages and categories
    """
    # Load messages dataset:
    messages = pd.read_csv(messages_filepath, sep=',')
    # Load categories dataset:
    categories = pd.read_csv(categories_filepath, sep=',')
    # Merge datasets:
    df = pd.merge(messages, categories, on="id")
    # Finally, return dataframe with merged messages and categories:
    return df
    

def clean_data(df):
    """ Function to clean the data that have been loaded before via function load_data.
        This function performs the following steps:
        1. Split categories into separate category columns.
        2. Convert category values to just numbers 0 or 1.
        3. Replace categories column in the DataFrame with new category columns.
        4. Remove duplicates from the dataframe
        
        Parameters: 
        - df (pandas.DataFrame): Dataframe which includes the merged messages and categories data.
        
        Returns: 
        - df (pandas.DataFrame): DataFrame that has been cleaned as described above
    """
    
    # STEP 1: Split categories into separate category columns:
    categories = df['categories'].str.split(pat=';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.loc[:0]
    
    # Extract a list of new column names for categories:
    category_colnames = []
    for x in range(0, 36): # Loop through columns...
        # Remove the first 4 (column index + 3 spaces) as well as the last two ('-1' or '-0') characters
        # from the string in column x:
        category_colnames.append(row[x].to_string()[5:len(row[x])-3]) 
    
    # Rename the columns of `categories`:
    categories.columns = category_colnames
    
    # STEP 2: Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1] # str[-1] = take only the last character of the string
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # In categories, replace value "2" by value "1" (obviously data errors in the set, in category "related")
    categories.replace(2, 1, inplace=True)
    
    # STEP 4: Replace categories column in df with new category columns.
    # Drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # Concatenate the original dataframe with the new `categories` dataframe:
    df = pd.concat([df, categories], axis=1)
    
    # STEP 5: Remove duplicates:
    # Determine duplicates and store in temporary column:
    df['dup'] = df.duplicated()
    # Drop duplicates:
    df = df.drop(df[df.dup == True].index, axis=0)
    # Drop temporary column:
    df = df.drop(columns='dup', axis=1)

    # Finally, return cleaned dataframe:
    return df


def save_data(df, database_filename):
    """ Function that saves the loaded and cleaned in the DataFrame to a SQLite database.
        
        Parameters: 
        - df (pandas.DataFrame): Dataframe which includes the loaded and cleaned data.
        - database_filename (string): Relative filename (path + database name) of the database where the data should be stored.
        
        Returns: 
        - Nothing, just stores the data for further use.
    """
    
    # Save the clean dataset into an sqlite database:
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    
    return  


def main():
    """ Main function of process data. 
        Includes the workflow for load, cleaning and storing the data to a database.
        
        Parameters:
        No input or output parameters included, but has to be called with the following 
        arguments in the calling string:
        - filepath of the message file (disaster_messages.csv), 
        - filepath of the categories file (disaster_categories.csv),
        - filepath of the SQLite database where the loaded and cleaned data should be stored.
        
        Returns:
        - Nothing
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()