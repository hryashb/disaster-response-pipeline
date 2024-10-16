# import libraries
import pandas as pd 
from sqlalchemy import create_engine
import re
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt_tab')

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_messages', engine)
    X = df.message
    Y = df.drop(columns = ['message', 'original', 'genre'])
    #  removing columns with every entry as 0
    Y = Y.loc[:,~(Y.sum() == 0).to_numpy()]
    #  removing columns with every entry as 1
    Y = Y.loc[:, ~(Y.sum()==len(Y)).to_numpy()]


    return X,Y, Y.columns


def tokenize(text: str):
    """
    Tokenises input text, returning a list of clean tokens for each message.
    Inputs: text: str, a message
    Returns: a list of tokens for this message
    """
    # Replace all urls with a urlplaceholder string
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')

    text = re.sub(r"[^\w\s]"," ", text)
    
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower().strip() for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Constructs the model using an sklearn pipeline and gridsearch
    """
    pipeline  = Pipeline(steps=[
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(GradientBoostingClassifier()))
    ])

    # Define a parameter grid to search over
    param_grid = {
    # Parameters for CountVectorizer
    'vect__max_df': [0.75, 0.9],  # Ignore words that appear in more than 75% or 90% of documents
    'vect__ngram_range': [(1, 1), (1, 2)],  # Use unigrams and bigrams
    
    # Parameters for GradientBoostingClassifier inside MultiOutputClassifier
    'moc__estimator__n_estimators': [50, 100],  # Number of boosting stages
    'moc__estimator__learning_rate': [0.1, 0.01],  # Learning rate for boosting
    'moc__estimator__max_depth': [3, 5],  # Maximum depth of trees
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=3, n_jobs=1)

    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a multi-output classifier using precision, recall, f1, and accuracy metrics.
    
    Parameters:
    model: sklearn model
    X_test: X test data
    y_test array: True labels, shape (n_samples, n_outputs)
    
    Returns:
    dict: A dictionary with average precision, recall, f1, and accuracy across all outputs.
    """
    #
    y_pred = model.predict(X_test)

    # Initialize lists to store scores for each output
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []

    # Iterate over each output (column in y_test and y_pred)
    for i in range(y_test.shape[1]):
        # For each output label, compute precision, recall, f1, and accuracy
        precision = precision_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
        recall = recall_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
        accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        
        # Append to respective lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)

    # Calculate average scores across all outputs
    average_precision = sum(precision_list) / len(precision_list)
    average_recall = sum(recall_list) / len(recall_list)
    average_f1 = sum(f1_list) / len(f1_list)
    average_accuracy = sum(accuracy_list) / len(accuracy_list)

    # Return results as a dictionary
    return {
        'average_precision': average_precision,
        'average_recall': average_recall,
        'average_f1': average_f1,
        'average_accuracy': average_accuracy
    }



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        best_model = model.best_estimator_
        print("Best parameters found: ", model.best_params_)
        print("Best cross-validation score: {:.3f}".format(model.best_score_))
        
        print('Evaluating model...')
        model_evaluation = evaluate_model(best_model, X_test, Y_test, category_names)
        print(model_evaluation)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
