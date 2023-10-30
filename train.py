import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.regularizers import l2
import requests

with open('data.json', 'r') as f:
    data = json.load(f)

job_types = [job["job_type"] for job in data["jobs"]]
unique_job_types = list(set(job_types))
user_interests = [interest for job in data['jobs'] for interest in job['user_interests']]

job_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
user_interest_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

job_encoder.fit(np.array(job_types).reshape(-1, 1))
user_interest_encoder.fit(np.array(user_interests).reshape(-1, 1))

def encode_interests(interests, encoder):
    interests_array = np.array(interests).reshape(-1, 1)
    return encoder.transform(interests_array).sum(axis=0)

X = []
y = []

for job in data['jobs']:
    user_interests_encoded = encode_interests(job['user_interests'], user_interest_encoder)
    job_type_encoded = job_encoder.transform(np.array([job['job_type']]).reshape(-1, 1)).flatten()
    X.append(user_interests_encoded)
    y.append(job_type_encoded)

X = np.array(X)
y = np.array(y)

jitter_amount = 0.01
X_jittered = X + np.random.normal(0, jitter_amount, X.shape)
X = np.vstack([X, X_jittered])
y = np.vstack([y, y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(unique_job_types), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}, Model Accuracy: {accuracy}")

model.save("model")


user_interests_to_predict = ["education", "mentoring", "working_with_children"]
user_interests_encoded = encode_interests(user_interests_to_predict, user_interest_encoder)

predictions = model.predict(user_interests_encoded.reshape(1, -1))
predicted_job_type = unique_job_types[np.argmax(predictions)]

print(f"Predicted Job Type: {predicted_job_type}")

url = 'http://localhost:8000/predict/8237823057802'

data = {
    'user_interests': ["coding", "creating", "work_life_balance"],
    'user_interest_encoder': ' '.join(map(str, user_interests_encoded))
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print('Request was successful.')
    print('Response content:')
    print(response.json())
else:
    print('Request failed with status code:', response.status_code)
    print('Response content:')
    print(response.text)