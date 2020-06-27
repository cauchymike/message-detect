from flask import Flask, render_template, request
import pickle
import numpy as np


#load the vectorizer and classifier
filename='cv-transform.pkl'
cv = pickle.load(open(filename, 'rb'))

classifier = pickle.load(open('my_classi.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/result1', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message] #put the message as a list
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        if int(my_prediction) == 1:
            my_prediction = "WARNING! This is a SPAM message!"
        else:
            my_prediction = "This is Not a spam Message."
        return  render_template('result1.html', my_prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)





