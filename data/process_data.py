import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    # Drop drop_duplicates
    df.drop_duplicates(inplace=True)

    # Split categories column
    categories = df.categories.str.split(";",expand=True)

    # Extract columns names from the first row
    categories.columns = categories.iloc[0,].str.split("-",expand=True)[0]

    # Keep only 0/1 from each value
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Remove the previous categories column and replace with the new ones
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df,categories], axis=1)

    return df

def save_data(df, database_filename):
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False)


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
