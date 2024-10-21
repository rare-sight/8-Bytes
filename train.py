import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

data_dir = 'gesture_data'
combined_data = []

for gesture_label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, gesture_label)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('_data.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path, header=None)
                df['label'] = gesture_label
                combined_data.append(df)

if combined_data:
    combined_df = pd.concat(combined_data, ignore_index=True)
else:
    print("No gesture data found to combine.")
    exit()

X = combined_df.drop('label', axis=1).values
y = combined_df['label'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

with open('svm_gesture_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("SVM model and label encoder saved.")
