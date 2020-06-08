from flask import Flask,render_template,url_for,request
from tensorflow.keras.models import load_model
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

filename = 'nlp_model.h5'
clf = load_model(filename)
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vect = preprocess_transform(message)
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

def preprocess_transform(message):
    ps = PorterStemmer()
    corpus = []
    vocab_size = 10000
    sent_len = 20
    
    msg = re.sub('[^A-Za-z]', ' ', message)
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word) for word in msg if word not in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg) 
    
    one_hot_rep = [one_hot(word, vocab_size) for word in corpus]
    
    embedded = pad_sequences(one_hot_rep,maxlen=sent_len)
    
    return np.array(embedded)

if __name__ == '__main__':
	app.run(debug=True, port=2020)