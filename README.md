# LSTM Next Word Prediction - "The Adventures of Sherlock Holmes" by Arthur Conan Doyle
## Overview
This project demonstrates the use of Long Short-Term Memory (LSTM) networks to predict the next word in a given text. The model is trained on Gutenberg's "The Adventures of Sherlock Holmes" by Arthur Conan Doyle, a public domain book. The goal of the project is to predict the next word in a sequence of text based on the context provided by the previous words. The model is trained for 100 epochs to improve its performance in word prediction.

## Project Goals
Text Preprocessing: Tokenize and prepare the text data for training.
LSTM Model: Build and train a deep learning model based on the LSTM architecture.
Next Word Prediction: Use the trained model to predict the next word in a sequence of text.
Epochs: Train the model for 100 epochs to optimize the weights and enhance accuracy.
 
 ## Dataset
The dataset used for this project is Gutenberg's "The Adventures of Sherlock Holmes" by Arthur Conan Doyle, which is publicly available and freely accessible. The text is split into sentences, and each sentence is used as a sequence for training the LSTM model. The model learns to predict the next word based on the sequence of words in the text.

You can access the full text from the following link: The Adventures of Sherlock Holmes on Project Gutenberg

## Model Architecture
The model used for next-word prediction is a Sequential LSTM network. The architecture is designed to learn the patterns and dependencies in the sequence of words from the text, enabling it to predict the next word in a given sentence. Key layers used in the architecture include:

Embedding Layer: Converts words into dense vectors of fixed size.
LSTM Layer: A recurrent layer that learns the temporal dependencies in the text.
Dense Layer: Outputs the prediction for the next word.
**Parameters**:
Epochs: 100
Batch Size: 64 (adjustable based on available memory)
Vocabulary Size: The unique words in the book are considered for training.
Sequence Length: A sequence of words is considered as input for the LSTM.
## Training
The LSTM model is trained on the tokenized text data for 100 epochs, with each epoch improving the model's accuracy in predicting the next word in the sequence. The training process involves:

Data Preprocessing: Tokenizing the text and converting it into sequences of words.
Model Training: Feeding the sequences to the LSTM model for learning the context and dependencies in the data.
Loss and Accuracy: The loss function is optimized to minimize the difference between predicted and actual words, and accuracy is calculated to measure prediction performance.
Training Command:
python
Copy code
model.fit(X_train, y_train, epochs=100, batch_size=64)
## Usage
Once the model is trained, you can use it to predict the next word in a sequence of words from the book. The user provides a sequence of words, and the model predicts the next word based on the patterns learned from the training data.

Example Usage:
python
Copy code
**input_text = "Sherlock Holmes sat in his chair"
predicted_word = predict_next_word(input_text)
print("Predicted next word:", predicted_word)**
This will output the most likely next word after the given input text based on the model's predictions.

## Installation
To run this project, you need to have the following libraries installed:

TensorFlow (for building and training the LSTM model)
Keras (high-level API for TensorFlow)
NumPy (for numerical operations)
NLTK (for tokenizing the text)
Matplotlib (for visualizing training progress)
You can install the required libraries using the following:

bash
Copy code
**pip install tensorflow keras numpy nltk matplotlib**
## Results and Visualizations
During training, the model's performance is evaluated using accuracy and loss metrics, and the training progress can be visualized using Matplotlib.

Example Visualization:
python
Copy code
**import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend(['Accuracy', 'Loss'])
plt.show()**
This will plot the training accuracy and loss over the course of 100 epochs.

## Conclusion
This project demonstrates the ability of LSTM networks to learn from a text corpus and predict the next word in a sequence, specifically using "The Adventures of Sherlock Holmes" by Arthur Conan Doyle. By training the model for 100 epochs, it becomes proficient in predicting the next word based on context, showcasing the power of deep learning for text generation tasks.
