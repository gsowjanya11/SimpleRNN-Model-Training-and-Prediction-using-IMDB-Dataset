# IMDB Sentiment Analysis with Simple RNN and Streamlit

## 📌 Project Overview
This project demonstrates sentiment analysis on the **IMDB movie reviews dataset** using TensorFlow/Keras.  
We train a **Simple RNN model** with an **Embedding layer** and **padding sequences** to handle variable‑length text.  
Finally, we deploy the model as an interactive **Streamlit app** where users can input their own text and receive a **sentiment prediction (positive/negative)** along with a confidence score with a pretrained Simple RNN pickle file

---

## ⚙️ Features
- Load IMDB dataset directly from TensorFlow/Keras.
- Preprocess text:
  - Tokenization using IMDB word index.
  - Embedding layer for dense vector representation.
  - Padding sequences to fixed length.
- Train a Simple RNN model for binary sentiment classification.
- Deploy with Streamlit for interactive user input and prediction.

---

## 📂 Project Structure
├── main.py               # Streamlit app
├── Simple_RNN_Model_Training_using_IMDB_Dataset.ipynb             # Model training script
├── Embedding_the_text.ipynb        # Preprocessing functions
├── requirements.txt     # Dependencies
├── Predict.ipynb     # Dependencies

