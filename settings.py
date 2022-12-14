import tensorflow as tf


TRAINING_SETTINGS = {
    'model_1': {
        'name': 'model_1',
        'vocab_size': 10000,
        'max_length': 280,
        'epochs': 5,  # Less epochs for faster training
        'layers': [
            tf.keras.layers.Embedding(10000, 20),
            tf.keras.layers.LSTM(15, dropout=0.5),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    },
    'model_2': {
        'name': 'model_2',
        'vocab_size': 10000,
        'max_length': 280,
        'epochs': 5,
        'layers': [
            tf.keras.layers.Embedding(10000, 40, input_length=280),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(20, dropout=0.6)
            ),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    },
    'model_3': {
        'name': 'model_3',
        'vocab_size': 10000,
        'max_length': 280,
        'epochs': 5,
        'layers': [
            tf.keras.layers.Embedding(10000, 16, input_length=280),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    }
}
