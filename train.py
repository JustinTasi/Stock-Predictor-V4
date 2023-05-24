import os
import signal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, GRU, Conv1D, MaxPooling1D, Attention, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error

print("TensorFlow version:", tf.__version__)

# Define custom Metric
def accuracy(y_true, y_pred):
    acc = tf.reduce_mean(tf.abs((y_true / y_pred) * 100))

    if tf.greater(acc, 100):
        acc = tf.constant(0, dtype=tf.float32)

    return acc

# Define reward function
def get_reward(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    reward = (1 - mape)
    return reward

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Close', 'Adj Close', 'Volume', 'High', 'Low', 'SMA', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'supertrend_signal', 'RSI', 'aroon_up', 'aroon_down', 'kicking', 'upper_band_supertrend', 'lower_band_supertrend']])

# Split data into train and test sets
train_data_norm = data_norm[:int(0.8 * len(data))]
test_data_norm = data_norm[int(0.8 * len(data)):]

# Define time steps
timesteps = 100

# Create sequences of timesteps
def create_sequences(data, timesteps):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data_norm, timesteps)
X_test, y_test = create_sequences(test_data_norm, timesteps)

# Build model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(LSTM(units=400, return_sequences=True, input_shape=(timesteps, X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(GRU(units=400, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=400))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(units=400)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(units=200, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(GRU(units=200, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(units=200)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(units=150))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=1))

# Compile model with MAPE loss and accuracy metric
learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mae", metrics=[accuracy])

# Function to handle SIGINT signal (CTRL + C)
def handle_interrupt(signal, frame):
    print("\nInterrupt received. Evaluating the Model and ending program...")
    # Perform the necessary actions before ending the program
    
    # Evaluate model
    model = load_model("model.h5")
    print("\nTest 1\n")
    y_pred_test = model.predict(X_test)
    test_reward = get_reward(y_test, y_pred_test)

    # Get accuracy
    print("\nTest 2\n")
    _, test_accuracy = model.evaluate(X_test, y_test)

    print("Test reward:", test_reward)
    print("Test accuracy:", test_accuracy)

    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

# Define callbacks
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train model
history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

# Evaluate model
model = load_model("model.h5")
print("\nTest 1\n")
y_pred_test = model.predict(X_test)
test_reward = get_reward(y_test, y_pred_test)

# Get accuracy
print("\nTest 2\n")
_, test_accuracy = model.evaluate(X_test, y_test)

print("Test reward:", test_reward)
print("Test accuracy:", test_accuracy)
