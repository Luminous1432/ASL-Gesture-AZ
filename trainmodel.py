from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import os
import numpy as np

label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Loop through actions and sequences to load data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                # Load the .npy file for the current frame
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            except FileNotFoundError:
                print(f"File not found: {os.path.join(DATA_PATH, action, str(sequence), '{}.npy'.format(frame_num))}")
                continue
        if len(window) == sequence_length:  # Ensure the sequence is complete
            sequences.append(window)
            labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Summarize the model structure
model.summary()

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
