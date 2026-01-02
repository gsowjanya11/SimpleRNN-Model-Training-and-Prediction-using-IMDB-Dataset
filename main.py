# Import Libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


# Import Pre-Trained Simple RNN Model
model=load_model('simple_rnn_imdb.h5')

# Function to preprocess the text and predict the score
word_index=imdb.get_word_index()
def preprocess(text):
  words=text.lower().split()
  encoded_text=[word_index.get(word,2)+3 for word in words]
  padded_text=pad_sequences([encoded_text],maxlen=500,padding='pre')
  return padded_text

def prediction(review):
  preprocessed_review=preprocess(review)
  predicted_review=model.predict(preprocessed_review)
  sentiment= "Positive" if predicted_review[0][0]>0.8 else "Negative"
  return sentiment,predicted_review[0][0]

# Streamlit Interface for user to input the movie review

# Streamlit app

import streamlit as st
st.title("Movie Prediction")
user_input=st.text_input("Enter a movie review to predict the sentiment")

if st.button("classify"):
  preprocess(user_input)
  sentiment,score=prediction(user_input)

  # Display the result
  st.write("The sentiment for given movie review is :", sentiment)
  st.write("The score for given movie review is :", score)

