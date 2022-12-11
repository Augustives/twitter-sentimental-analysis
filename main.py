
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from utils import (
    depure_data, sent_to_words, detokenize,
    remove_stop_words, nltk_setup
)
from settings import TRAINING_SETTINGS

# Config
nltk_setup()
MODEL_NAME = 'model_2'
training_setting = TRAINING_SETTINGS[MODEL_NAME]


# Prepare data
data = pd.read_csv('./data/treated_data.csv')
df = pd.DataFrame(data, columns=['label', 'tweet'])

df['tweet'] = df['tweet'].apply(remove_stop_words)
temp = []
data_to_list = df['tweet'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
data_words = list(sent_to_words(temp))

treated_tweets = []
for i in range(len(data_words)):
    treated_tweets.append(detokenize(data_words[i]))


# Train model
tokenizer = Tokenizer(num_words=training_setting['vocab_size'])
tokenizer.fit_on_texts(treated_tweets)
sequences = tokenizer.texts_to_sequences(treated_tweets)
tweets = pad_sequences(sequences, maxlen=training_setting['max_length'])

X_train, X_test, y_train, y_test = train_test_split(
    tweets, df['label'], random_state=0
)

if not os.path.exists(f'./models/best_{training_setting["name"]}.hdf5'):
    print('Training in progress')
    model = tf.keras.Sequential(training_setting['layers'])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    checkpoint = ModelCheckpoint(
        f'./models/best_{training_setting["name"]}.hdf5',
        monitor='val_accuracy',
        verbose=1, save_best_only=True, mode='auto',
        period=1, save_weights_only=False
    )
    history = model.fit(
        X_train, y_train, epochs=training_setting['epochs'],
        validation_data=(X_test, y_test), callbacks=[checkpoint]
    )


# Analyze model
best_model = keras.models.load_model(
    f'./models/best_{training_setting["name"]}.hdf5'
)
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
predictions = best_model.predict(X_test)
predictions_binary = [
    np.where(prediction > 0.75, 1, 0).max()
    for prediction in predictions
]

sentiment = ['Non Offensive', 'Offensive']
matrix = confusion_matrix(
    y_test,
    predictions_binary,
)
conf_matrix = pd.DataFrame(
    matrix,
    index=sentiment,
    columns=sentiment
)
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='g')


# Avaliation on tweet users
def avaliate_tweet(tweet):
    sequence = tokenizer.texts_to_sequences([tweet])
    test = pad_sequences(sequence, maxlen=training_setting['max_length'])
    return sentiment[np.where(best_model.predict(test) > 0.75, 1, 0)[0][0]]


avaliate_tweet("this bitch sucks")
avaliate_tweet("good job mate")

for (dirpath, dirnames, filenames) in os.walk('./data/tweet_accs'):
    tweet_avaliation = {}
    for filename in filenames:
        data = pd.read_csv(f'./data/tweet_accs/{filename}')
        df = pd.DataFrame(data, columns=['Tweet'])
        df = df.dropna()
        results = [
            avaliate_tweet(tweet)
            for tweet in df['Tweet'].to_list()
        ]
        tweet_avaliation[filename.split('.')[0]] = (
            round(results.count('Offensive') / len(results), 4) * 100
        )

for key, value in tweet_avaliation.items():
    print(f'User {key} has {value}% of his tweets considered offensive!\n')
