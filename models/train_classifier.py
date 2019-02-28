import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, \
    f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
        Load data from sqlite database
        Arguments:
            database_filepath: path to database file
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    sql = 'SELECT * FROM DisasterPipeline'
    df = pd.read_sql(sql, engine)
    x = df.message
    y = df.iloc[:, 4:]
    y_labels = list(y)

    return x, y, y_labels


def tokenize(text):
    """ Tokenize text
        Arguments:
            text: message string
        Returns:
            tokens: okenized text
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return tokens


def build_model():
    """ Build ML pipeline

        Returns:
            cv: GridSearchCV object
    """
    num_thread = 1

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, min_df=5)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=10,
                min_samples_split=10),
                n_jobs=num_thread
            )
        )
    ])

    parameters = {
        'vect__min_df': [2, 4],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 50],
        'clf__estimator__min_samples_split':[2, 4, 8]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10,
                      n_jobs=num_thread)
    return cv


def display_results(labels, pred, col):
    """ Display model metrics based on the model predictions

        Arguments:
            labels: data labels
            pred: predicted labels
            col: list of column names
    """
    results = []

    for i in range(len(col)):
        accuracy = accuracy_score(labels[:, i], pred[:, i])
        precision = precision_score(labels[:, i], pred[:, i], average="micro")
        recall = recall_score(labels[:, i], pred[:, i], average="micro")
        f1 = f1_score(labels[:, i], pred[:, i], average="micro")

        results.append([accuracy, precision, recall, f1])

    print(results)


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate model

        Arguments:
            model: model
            X_test: test dataset
            Y_test: test labels
            category_names: category name list
    """
    pred = model.predict(X_test)

    display_results(np.array(Y_test), pred, category_names)


def save_model(model, model_filepath):
    """ Save fitted model

        Arguments:
            model: model
            model_filepath: file path
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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