this model version, model5, is a restart on the machine learning work
was gone a month and now I don't remember anything, will rebuild from the ground up


------
Training a Classifier:

Collect data for each gesture by recording the landmarks and corresponding labels.
Train a classifier (e.g., SVM, Random Forest, or a neural network) on this data.
python

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Example feature extraction function
def extract_features(landmarks):
    # Calculate distances and angles, normalize, etc.
    features = []
    # Add custom feature extraction logic here
    return np.array(features)

# Prepare your dataset
X = []  # Feature vectors
y = []  # Corresponding labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


Real-Time Gesture Recognition:

Use the trained classifier to predict gestures in real-time.
Integrate this into the video processing loop, where you extract features from the current frame and pass them to the classifier.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            features = extract_features(landmarks)
            gesture = classifier.predict([features])
            # Display the gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()




okay so new idea

we use csv files to store the dynamic gesture points of a period of time - temporal data

we can save data into a tree structure, like that of the Mediapipe demo but instead of images we have multiple csv files


so after doing a test run of the csv, we get a lot of data points, might need to do something about this, like put labels on the data or something
stuff to do
- create a heading for the csv data
- save only the landmark position, the 'test1.csv' has alot of stuff in it, might be wrong though as i counted 63 instead of 60 columns
- we need features and outputs
- somehow build a temporal model out of the csv



,frame,x_0,y_0,z_0,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5,x_6,y_6,z_6,x_7,y_7,z_7,x_8,y_8,z_8,x_9,y_9,z_9,x_10,y_10,z_10,x_11,y_11,z_11,x_12,y_12,z_12,x_13,y_13,z_13,x_14,y_14,z_14,x_15,y_15,z_15,x_16,y_16,z_16,x_17,y_17,z_17,x_18,y_18,z_18,x_19,y_19,z_19,x_20,y_20,z_20,frame_rate,frame_width,frame_height,gesture,gesture_index,distance_0_1


frame,x_0,y_0,z_0,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5,x_6,y_6,z_6,x_7,y_7,z_7,x_8,y_8,z_8,x_9,y_9,z_9,x_10,y_10,z_10,x_11,y_11,z_11,x_12,y_12,z_12,x_13,y_13,z_13,x_14,y_14,z_14,x_15,y_15,z_15,x_16,y_16,z_16,x_17,y_17,z_17,x_18,y_18,z_18,x_19,y_19,z_19,x_20,y_20,z_20,frame_rate,frame_width,frame_height,gesture,gesture_index,distance_0_1
