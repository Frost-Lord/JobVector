import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.regularizers import l2
import requests

def load_data():
    with open('data.json', 'r') as f:
        return json.load(f)

def get_encoders(data):
    job_types = [job["job_type"] for job in data["jobs"]]
    user_interests = list(set(interest for job in data['jobs'] for interest in job['user_interests']))

    job_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    user_interest_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    job_encoder.fit(np.array(job_types).reshape(-1, 1))
    user_interest_encoder.fit(np.array(user_interests).reshape(-1, 1))

    print(f"Job Types: {job_encoder.categories_[0]}")
    print(f"User Interests: {user_interest_encoder.categories_[0]}")
    print(f"Number of Job Types: {len(job_encoder.categories_[0])}")
    print(f"Number of User Interests: {len(user_interest_encoder.categories_[0])}")
    
    
    return job_encoder, user_interest_encoder

def encode_interests(interests, encoder):
    interests_array = np.array(interests).reshape(-1, 1)
    return encoder.transform(interests_array).sum(axis=0).reshape(1, -1)

def prepare_data(data, job_encoder, user_interest_encoder):
    X, y = [], []

    for job in data['jobs']:
        user_interests_encoded = encode_interests(job['user_interests'], user_interest_encoder)
        job_type_encoded = job_encoder.transform(np.array([job['job_type']]).reshape(-1, 1))
        X.append(user_interests_encoded)
        y.append(job_type_encoded)

    X, y = np.vstack(X), np.vstack(y)
    jitter_amount = 0.01
    X_jittered = X + np.random.normal(0, jitter_amount, X.shape)
    
    X = np.vstack([X, X_jittered])
    y = np.vstack([y, y])

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(input_shape, output_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])

def main():
    data = load_data()
    job_encoder, user_interest_encoder = get_encoders(data)
    X_train, X_test, y_train, y_test = prepare_data(data, job_encoder, user_interest_encoder)

    model = create_model((X_train.shape[1],), y_train.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Loss: {loss}, Model Accuracy: {accuracy}")

    model.save("model")

    user_interests_to_predict = ["education", "mentoring", "working_with_children"]
    user_interests_encoded = encode_interests(user_interests_to_predict, user_interest_encoder)

    predictions = model.predict(user_interests_encoded)
    unique_job_types = list(job_encoder.categories_[0])
    predicted_job_type = unique_job_types[np.argmax(predictions)]
    print(f"Predicted Job Type: {predicted_job_type}")

    url = 'http://localhost:8000/predict/8237823057802'
    response_data = {
        'user_interests': ["coding", "creating", "work_life_balance"],
        'user_interest_encoder': ' '.join(map(str, user_interests_encoded.flatten()))
    }

    response = requests.post(url, json=response_data)
    if response.status_code == 200:
        print('Request was successful.')
        print('Response content:')
        print(response.json())
    else:
        print('Request failed with status code:', response.status_code)
        print('Response content:')
        print(response.text)

if __name__ == "__main__":
    main()