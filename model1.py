import numpy as np
import pandas as pd
import pickle


#load the data with a beautiful function
def load_data(data_input):
    train_data = pd.read_csv(data_input, encoding='latin-1')
    train_data = train_data.dropna(axis=1, how='any')
    train_data.columns = ['label', 'message']
    train_data['label_num'] = train_data.label.map({'ham': 0, 'spam': 1})

    return train_data

#create an instance
train_data=load_data('datasets_483_982_spam.csv')





#lets import libraries for NLP
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer  #we use stemmer to handlewords that mean the same thing in a sentence

#lets clean the sms with another beautiful function

import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train_data['clean_message']= train_data.message.apply(clean_text_round1)


def remove_stop(mess_clean):
    """define the stop words,
    we will be adding additional stopwords
    """
    stop = stopwords.words('english') + ['u', 'ur','4', '2', 'im','dont', 'doin','ure'] # stopwords was defined when we initiated the word cloud

    return ' '.join([word for word in mess_clean.split() if word not in stop])  # WE CAN ALSO USE THE LAMBDA FUNCTION

train_data['clean_message']= train_data.clean_message.apply(remove_stop)



#now, lets split our entire data into messages and labels with a simple function
from sklearn.feature_extraction.text import CountVectorizer

# create an instance of the countvectorizer
cv = CountVectorizer()
X= cv.fit_transform(train_data['clean_message']).toarray()
y = train_data.label_num
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



#building our model

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.4)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn import metrics
metrics.confusion_matrix(y_test, y_pred)


#lets create a pickle for both cv and classifier

filename='cv-transform.pkl'
pickle.dump(cv, open(filename, 'wb'))


pickle.dump(classifier, open('my_classi.pkl', 'wb'))






