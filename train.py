import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

with open('data.json', 'r') as f:
    data = json.load(f)

items = ["apple", "banana", "carrot", "donut", "egg", "fish"]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(items).reshape(-1, 1))

def encode_cart(cart):
    cart_items = np.array(list(cart.keys())).reshape(-1, 1)
    return encoder.transform(cart_items).sum(axis=0)

X = []
y = []


for user in data['users']:
    current_cart_encoded = encode_cart(user['current_cart'])
    for previous_cart in user['previous_carts']:
        X.append(encode_cart(previous_cart))
        y.append(current_cart_encoded)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(items), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}, Model Accuracy: {accuracy}")

model.save("saved_model")