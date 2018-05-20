#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model

from keras.preprocessing import sequence
from keras.layers import Dense, Conv2D, Dropout, Embedding, LSTM, Bidirectional
from keras.activations import relu, tanh, sigmoid, elu
from keras.models import Sequential

''' Multivariate Bi-Directional LSTM Model,
try to remember for 3,280 succesive iterations
Or cluster based on eras?? '''
def build_nn(x_train, y_train):

    XX = np.array(x_train)
    YY = np.array(y_train)

    data_dim = 50 #We have 50 feautures max_len
    timesteps = 3200 #?? For each era -- think about a better way of doing this max_features
    num_classes = 2
    batch_size = 32

    model = Sequential()
    model.add(Embedding(timesteps, 128, input_length=data_dim))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(LSTM(32, return_sequences=True, input_shape = (timesteps, data_dim)))
    # model.add(LSTM(32))
    # model.add(Dense(activation = 'softmax')

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    print 'Training...'
    model.fit(XX, YY, batch_size = 32, epochs = 5)

def train(model, x_train, y_train):

    model.fit(x_train, y_train, batch_size = 32, epocsh = 5)

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)


    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    # This is your model that will learn to predict
    print("Training...")
    # Your model is trained on the training_data
    build_nn(X, Y)

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
