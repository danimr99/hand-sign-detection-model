import pickle
import cv2
import mediapipe as mp
import numpy as np

import tensorflow as tf
import tensorflow_decision_forests as tfdf

# Definition of constants
DATASET_FILE_PATH = '../../dataset/dataset.pickle'
MODEL_FILE_PATH = '../../models/tensorflow/model'
MIN_PREDICTION_CONFIDENCE = 0.7
CONSECUTIVE_PREDICTIONS_FRAMES = 10

# Load data
data_dict = pickle.load(open(DATASET_FILE_PATH, 'rb'))

# Get data and labels
labels = np.asarray(data_dict['labels'])

# Convert each label into an integer identifier depending on the label (same label = same identifier)
unique_labels = np.unique(labels)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
labels = np.asarray([label_to_id[label] for label in labels])

# Load model
model = tf.keras.models.load_model(MODEL_FILE_PATH)

# Definition of variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up mediapipe instance
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# Read webcam
cap = cv2.VideoCapture(0)

# Detect from webcam
predictions = []

while True:
    aux = []
    x_ = []
    y_ = []

    # Read webcam frame
    ret, frame = cap.read()

    H, W, _ = frame.shape

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detection
    results = hands.process(frame_rgb)

    # Get landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                aux.append(x - min(x_))
                aux.append(y - min(y_))

            # Append zeros if sign uses one hand only
            if len(results.multi_hand_landmarks) == 1:
                for i in range((21 * 2)):
                    aux.append(0)

        # Draw bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict
        prediction = model.predict_on_batch([np.asarray(aux)])

        # Get the maximum value in the prediction array
        prediction_confidence = np.max(prediction)

        # Get the position of the maximum value in the prediction array
        prediction_class = np.argmax(prediction, axis=1)[0]

        # Convert prediction class to label
        prediction_class = unique_labels[prediction_class]

        print(prediction_class, prediction_confidence)

        # Check prediction confidence
        if prediction_confidence >= MIN_PREDICTION_CONFIDENCE:
            # Add prediction to list if it is the same as last consecutive predictions
            if len(predictions) == 0 or prediction_class in predictions:
                predictions.append(prediction_class)
            else:
                # Clear predictions list if prediction is different from last consecutive predictions
                predictions.clear()
                predictions.append(prediction_class)

            # Draw prediction based on last consecutive predictions frames
            if len(predictions) >= CONSECUTIVE_PREDICTIONS_FRAMES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predictions[-1], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

    # Show webcam frame
    cv2.imshow('Live Demo', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
