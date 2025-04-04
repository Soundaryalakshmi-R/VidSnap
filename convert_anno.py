
import pandas as pd
import numpy as np
import cv2
import tensorflow_hub as hub
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# Load annotations
anno_df = pd.read_csv('/Users/apple/Desktop/datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv', sep='\t', header=None,
                      names=['shot_id', 'video_id', 'scores'])
anno_df['mean_score'] = anno_df['scores'].apply(lambda x: np.mean([int(s) for s in x.split(',')]))
print("First 5 rows of anno_df:")
print(anno_df.head())
print("Available shot IDs in annotations:", anno_df['shot_id'].unique())
# Load pretrained model
model = hub.load('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5')

def extract_features(video_path, sample_rate=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // sample_rate)  # Sample every 0.5s (2 fps)
    features = []
    frame_ids = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = preprocess_input(frame.astype(np.float32))
            frame = np.expand_dims(frame, axis=0)
            feature = model(frame)
            features.append(feature.numpy()[0])
            frame_ids.append(count)
        count += 1
    cap.release()
    return np.array(features), frame_ids

# Directory for saving features
output_dir = 'features'
os.makedirs(output_dir, exist_ok=True)

# Process all videos
video_dir = 'tvsum50_ver_1_1/ydata-tvsum50-v1_1/video'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
for video_file in video_files:
    video_id = video_file[:-4]  # Remove .mp4 extension
    video_path = os.path.join(video_dir, video_file)
    
    # Extract features
    features, frame_ids = extract_features(video_path)
    np.save(os.path.join(output_dir, f'features_{video_id}.npy'), features)
    np.save(os.path.join(output_dir, f'frame_ids_{video_id}.npy'), frame_ids)
    print(f"Extracted {len(features)} features for {video_id}")

    # Assign features to shots
    num_shots = len(anno_df[anno_df['shot_id'] == video_id])
    num_frames = len(features)
    if num_shots == 0:
        print(f"Warning: No shots found for shot_id '{video_id}' in annotations.")
        continue
    
    frames_per_shot = max(1, num_frames // num_shots)
    shot_features = []
    shot_scores = []
    for i in range(num_shots):
        start = i * frames_per_shot
        end = min((i + 1) * frames_per_shot, num_frames)
        shot_feats = features[start:end].mean(axis=0)
        shot_features.append(shot_feats)
        shot_scores.append(anno_df[anno_df['shot_id'] == video_id]['mean_score'].iloc[i])
    
    X = np.array(shot_features)  # Shape: (num_shots, 2048)
    y = np.array(shot_scores)    # Shape: (num_shots,)
    np.save(os.path.join(output_dir, f'shot_features_{video_id}.npy'), X)
    np.save(os.path.join(output_dir, f'shot_scores_{video_id}.npy'), y)
    print(f"Saved shot features for {video_id}: X shape {X.shape}, y shape {y.shape}")