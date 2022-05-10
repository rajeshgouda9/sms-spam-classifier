import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open('vect.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
ps = PorterStemmer()
en = stopwords.words('english')


def textprocess(x):
    x = x.lower()
    x = re.sub(r"[$&+,:;=?@#|'<>.-^*()%!]", ' ', x)
    x = nltk.word_tokenize(x)
    y = []
    for i in x:
        if i not in en and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)


st.title('SMS Spam Classifier')
input_sms = st.text_area('Enter the SMS')
if st.button('Predict'):

    # preprocess
    transform_sms = textprocess(input_sms)
    # vectorized
    vect_sms = tfidf.transform([transform_sms])
    # predict
    res = model.predict(vect_sms)
    if res == 0:
        st.header('Not Spam')
    else:
        st.header('Spam')
