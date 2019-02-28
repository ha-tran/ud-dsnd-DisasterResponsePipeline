import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load datasets and merge data
        Arguments:
            messages_filepath: file path for messages dataset
            categories_filepath: file path for categories dataset
        Returns:
            dataframe of merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, how='left', on=['id'])


def clean_data(df):
    """ Clean dataset
        Arguments:
            df: dataframe of dataset
        Returns:
            cleaned dataframe
    """
    # a copy of df with 36 category columns
    cat = df['categories'].str.split(';', expand=True)

    # get first row and extract a list of new category column names
    row = cat.iloc[0]
    cat_cols = row.transform(lambda x: x[:-2]).tolist()
    # assign new column names to cat dataframe
    cat.columns = cat_cols

    # Numerize category values
    for col in cat:
        cat[col] = pd.to_numeric(cat[col].str[-1])

    # Drop original categories
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, cat], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """ Clean dataset
        Arguments:
            df: dataframe
            database_filename: name of sqlite file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterPipeline', engine, index=False)


def main():
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