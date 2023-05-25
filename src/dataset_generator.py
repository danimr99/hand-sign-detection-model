import os
import pickle
import mediapipe as mp
import cv2

# Definition of constants
IMAGES_DIR_PATH = '../images'
EXCLUDED_FILES = ['.DS_Store']
OUTPUT_FILE_PATH = '../dataset/dataset.pickle'

# Definition of variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up mediapipe instance
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Read images from each class name directory
data = []
labels = []

for class_label in os.listdir(IMAGES_DIR_PATH):
    # Ignore .DS_Store
    if class_label not in EXCLUDED_FILES:
        for image_name in os.listdir(os.path.join(IMAGES_DIR_PATH, class_label)):
            aux = []
            x_ = []
            y_ = []

            # Read image
            image = cv2.imread(os.path.join(
                IMAGES_DIR_PATH, class_label, image_name))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Make detection
            results = hands.process(image_rgb)

            # Get landmarks
            if results.multi_hand_landmarks:
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

                data.append(aux)
                labels.append(class_label)

# Close mediapipe instance
hands.close()

# Create if not exists dataset directory from output file path
if not os.path.exists(os.path.dirname(OUTPUT_FILE_PATH)):
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH))

# Save data and labels
f = open(OUTPUT_FILE_PATH, 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
