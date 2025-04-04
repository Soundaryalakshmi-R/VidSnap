import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os

# Load trained model and scalers with custom objects
model = load_model('lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
scaler_X = np.load('scaler_X.npy', allow_pickle=True).item()
scaler_y = np.load('scaler_y.npy', allow_pickle=True).item()

# Load video shot features (example: AwmHb44_ouw)
video_id = 'AwmHb44_ouw'
X = np.load(f'features/shot_features_{video_id}.npy')  # Shape: (20, 2048)
X_normalized = scaler_X.transform(X)  # Normalize features

# Create sequences for prediction
timesteps = 10
X_seq = np.array([X_normalized[i:i+timesteps] for i in range(0, 20-timesteps+1)])  # Shape: (11, 10, 2048)

# Predict scores
y_pred = model.predict(X_seq)  # Shape: (11, 10, 1)
y_pred = y_pred.squeeze()     # Shape: (11, 10)

# Reconstruct full shot scores (average overlapping predictions)
y_full = np.zeros(20)
counts = np.zeros(20)
for i in range(11):
    y_full[i:i+10] += y_pred[i]
    counts[i:i+10] += 1
y_full /= counts  # Shape: (20,)
y_full = scaler_y.inverse_transform(y_full.reshape(-1, 1)).flatten()  # Back to original scale (e.g., 1-5)
print(f"Predicted scores for {video_id}: {y_full}")

# DPP for keyframe selection
def dpp_kernel(features, scores, num_keyframes=5):
    # Compute similarity matrix (cosine similarity)
    S = cosine_similarity(features)  # Shape: (20, 20)
    
    # Relevance scores (q)
    q = scores  # Shape: (20,)
    
    # Construct DPP kernel L = q[i] * S[i,j] * q[j]
    L = np.outer(q, q) * S  # Shape: (20, 20)
    
    # Greedy DPP selection
    selected = []
    remaining = list(range(len(features)))
    for _ in range(num_keyframes):
        if not remaining:
            break
        # Score each remaining item based on current L
        scores = [L[i, i] - sum(L[i, j]**2 / L[j, j] for j in selected if L[j, j] > 0) 
                 if i not in selected else -np.inf for i in remaining]
        next_idx = remaining[np.argmax(scores)]
        selected.append(next_idx)
        remaining.remove(next_idx)
    
    return selected

# Select keyframes
keyframe_indices = dpp_kernel(X, y_full, num_keyframes=5)
print(f"Selected keyframe indices for {video_id}: {keyframe_indices}")

# Load frame IDs for mapping
frame_ids = np.load(f'features/frame_ids_{video_id}.npy')
frames_per_shot = len(frame_ids) // 20  # Approximate
keyframe_frames = [frame_ids[min(i * frames_per_shot, len(frame_ids)-1)] for i in keyframe_indices]
print(f"Selected keyframe frame IDs for {video_id}: {keyframe_frames}")

model = load_model('lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
scaler_X = np.load('scaler_X.npy', allow_pickle=True).item()
scaler_y = np.load('scaler_y.npy', allow_pickle=True).item()

video_id = 'AwmHb44_ouw'
X = np.load(f'features/shot_features_{video_id}.npy')
X_normalized = scaler_X.transform(X)

timesteps = 10
X_seq = np.array([X_normalized[i:i+timesteps] for i in range(0, 20-timesteps+1)])

y_pred = model.predict(X_seq)
y_pred = y_pred.squeeze()

y_full = np.zeros(20)
counts = np.zeros(20)
for i in range(11):
    y_full[i:i+10] += y_pred[i]
    counts[i:i+10] += 1
y_full /= counts
y_full = scaler_y.inverse_transform(y_full.reshape(-1, 1)).flatten()
print(f"Predicted scores for {video_id}: {y_full}")

def dpp_kernel(features, scores, num_keyframes=5):
    S = cosine_similarity(features)
    q = scores
    L = np.outer(q, q) * S
    selected = []
    remaining = list(range(len(features)))
    for _ in range(num_keyframes):
        if not remaining:
            break
        scores = [L[i, i] - sum(L[i, j]**2 / L[j, j] for j in selected if L[j, j] > 0) 
                 if i not in selected else -np.inf for i in remaining]
        next_idx = remaining[np.argmax(scores)]
        selected.append(next_idx)
        remaining.remove(next_idx)
    return selected

keyframe_indices = dpp_kernel(X, y_full, num_keyframes=5)
print(f"Selected keyframe indices for {video_id}: {keyframe_indices}")

frame_ids = np.load(f'features/frame_ids_{video_id}.npy')
frames_per_shot = len(frame_ids) // 20
keyframe_frames = [frame_ids[min(i * frames_per_shot, len(frame_ids)-1)] for i in keyframe_indices]
print(f"Selected keyframe frame IDs for {video_id}: {keyframe_frames}")

# --- Step 5: Generate Video Summary ---
video_path = f'tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/{video_id}.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video
output_dir = 'summaries'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'summary_{video_id}.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Extract 1-second clips (30 frames) around each keyframe
clip_length = int(fps)  # 1 second of frames
for frame_id in keyframe_frames:
    start_frame = max(0, frame_id - clip_length // 2)  # Center clip around keyframe
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), start_frame + clip_length)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

cap.release()
out.release()
print(f"Video summary saved to {output_path}")