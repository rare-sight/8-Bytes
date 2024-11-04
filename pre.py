import pickle
import mediapipe as mp
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import os

with open('svm_gesture_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gesture_to_word = {
    'A': 'मदत',
    'B': 'शांती',
    'C': 'माझं तुझ्यावर प्रेम आहे',
    'D': 'उभा रहा',
    'E': 'बसा',
    'F': 'पाणी',
    'G': 'थांब',
    'H': 'मी सहमत आहे',
    'I': 'नमस्ते',
    'J': 'ठीक आहे'
}
output_file = 'recognized_gestures.txt'
last_gesture = None
last_time = time.time()
confidence_threshold = 0.9

font_path = "NotoSansDevanagari-Regular.ttf"
if not os.path.exists(font_path):
    print(f"Font file not found: {font_path}")
    exit(1)
font_size = 24
font = ImageFont.truetype(font_path, font_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
            keypoints = np.array(keypoints).reshape(1, -1)
            probabilities = svm_model.predict_proba(keypoints)[0]
            predicted_label_index = np.argmax(probabilities)
            predicted_label = le.inverse_transform([predicted_label_index])[0]
            confidence = probabilities[predicted_label_index]
            if predicted_label in gesture_to_word and confidence > confidence_threshold:
                word = gesture_to_word[predicted_label]
                draw.text((10, 30), f'Word: {word} ({confidence:.2f})', font=font, fill=(255, 0, 0))
                current_time = time.time()
                if predicted_label != last_gesture or (current_time - last_time) > 3:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"{word}\n")
                    last_gesture = predicted_label
                    last_time = current_time
            else:
                draw.text((10, 30), 'Word: Unknown', font=font, fill=(0, 0, 255))
    frame = np.array(pil_image)
    cv2.imshow("Hand Landmark Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
