import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

data_dir = "gesture_data"
os.makedirs(data_dir, exist_ok=True)

def extract_keypoints(hand_landmarks):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y])
    return np.array(keypoints)

def save_keypoints(keypoints, gesture_label):
    gesture_folder = os.path.join(data_dir, gesture_label)
    os.makedirs(gesture_folder, exist_ok=True)
    file_path = os.path.join(gesture_folder, f"{gesture_label}_data.csv")
    df = pd.DataFrame([keypoints])
    df.to_csv(file_path, mode='a', header=False, index=False)
    print(f"Saved keypoints for gesture: {gesture_label}")

def combine_gesture_data():
    combined_data = []
    print("Combining gesture data...")
    if not os.path.exists(data_dir):
        print("Data directory does not exist.")
        return
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        print(f"Checking folder: {folder_path}")
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('_data.csv'):
                    file_path = os.path.join(folder_path, file)
                    print(f"Reading file: {file_path}")
                    try:
                        df = pd.read_csv(file_path, header=None)
                        df['label'] = folder
                        combined_data.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv('combined_gesture_data.csv', index=False, header=False)
        print("Combined gesture data saved to combined_gesture_data.csv")
    else:
        print("No gesture data found to combine.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = extract_keypoints(hand_landmarks)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                save_keypoints(keypoints, 'A')
            elif key == ord('b'):
                save_keypoints(keypoints, 'B')
            elif key == ord('c'):
                save_keypoints(keypoints, 'C')
            elif key == ord('d'):
                save_keypoints(keypoints, 'D')
            elif key == ord('e'):
                save_keypoints(keypoints, 'E')
            elif key == ord('f'):
                save_keypoints(keypoints, 'F')
            elif key == ord('g'):
                save_keypoints(keypoints, 'G')
            elif key == ord('h'):
                save_keypoints(keypoints, 'H')
            elif key == ord('i'):
                save_keypoints(keypoints, 'I')
            elif key == ord('j'):
                save_keypoints(keypoints, 'J')
            
    cv2.imshow("Hand Landmark Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
combine_gesture_data()
