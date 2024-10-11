# Disaster Response Pipeline Project
This project deploys a Flask app which categorises messages into topics which they may be referring to. The purpose of this is to allow operators, automated or otherwise, to allocate response resources quickly and efficiently through the categorisation of these messages, removing time taken to interpret and process these messages.

## To get started: 

### Clone this repo using: 
`git clone XXX`

### CD into the project
`cd disaster_response_pipeline`

### Install requirements using 
`pip install -r requirements.txt`

### To deploy the app 
`python app/run.py`

### To access the app
Go to http://0.0.0.0:3001/

### To retrain the model: 
- Set up the database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- Train the model
`python models/train_classifier.py data/DisasterResponse.db models/bm.pkl`




## Details:

### 1. Tokenisation
- **Purpose**: Clean and tokenize text for classification.
- **Steps**:
  - Replace URLs with `'urlplaceholder'`.
  - Remove punctuation.
  - Tokenize text into words.
  - Remove stopwords.
  - Lemmatize words to their base form.

### 2. Pipeline:
- **Purpose**: Contains a series of steps which trains the model.
  1. CountVectorizer: Converts text to word counts using the tokenisation from the step above.
  2. TfidfTransformer: Convert counts to TF-IDF, which is a measure of how important a specific word is to a message with respect to the rest of the messages
  3. GradientBoostingClassifier: A classifier which uses a gradient boosting method.

### 3. Grid Search: 
- This iterates over parameters to train the model, finding the best set of parameters for this specific use case. 
