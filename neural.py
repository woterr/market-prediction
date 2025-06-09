import numpy as np
import tensorflow as tf
from data import load_data

epochs = 300
lr = 1e-2
treshold = 0.55

X_raw, Y= load_data(start="2024-01-01", end="2025-06-06")

norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(X_raw)
X = norm(X_raw)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid"),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.fit(X, Y, epochs=epochs)


X_test_raw, Y_test = load_data(start="2023-12-01", end="2023-12-10")
X_test = norm(X_test_raw)
y_new = model.predict(X_test)
y_pred_binary = (y_new >= 0.5).astype(int)
