import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- Step 2: Prepare Data ---
output_dir = 'features'
X_all = []
y_all = []
video_files = [f[:-4] for f in os.listdir('/Users/apple/Desktop/datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video') if f.endswith('.mp4')]
for video_id in video_files:
    X = np.load(os.path.join(output_dir, f'shot_features_{video_id}.npy'))
    y = np.load(os.path.join(output_dir, f'shot_scores_{video_id}.npy'))
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)
print(f"X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")

scaler_X = MinMaxScaler()
X_all_normalized = scaler_X.fit_transform(X_all)
scaler_y = MinMaxScaler()
y_all_normalized = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

def create_sequences(X, y, timesteps=10):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - timesteps + 1, timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i:i + timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 10
X_seq, y_seq = [], []
for i in range(0, len(video_files)):
    start_idx = i * 20
    end_idx = start_idx + 20
    X_video = X_all_normalized[start_idx:end_idx]
    y_video = y_all_normalized[start_idx:end_idx]
    X_s, y_s = create_sequences(X_video, y_video, timesteps)
    X_seq.append(X_s)
    y_seq.append(y_s)

X_seq = np.vstack(X_seq)
y_seq = np.vstack(y_seq)
print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('scaler_X.npy', scaler_X)
np.save('scaler_y.npy', scaler_y)

# --- Step 3: Build and Train LSTM ---
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Reshape y to match LSTM output
y_train_reshaped = y_train[..., np.newaxis]  # Shape: (80, 10, 1)
y_val_reshaped = y_val[..., np.newaxis]      # Shape: (20, 10, 1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(10, 2048), return_sequences=True))
model.add(Dense(1))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_reshaped,
    validation_data=(X_val, y_val_reshaped),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('lstm_model.h5')
print("LSTM model saved to 'lstm_model.h5'")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()