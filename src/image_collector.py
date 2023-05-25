import os
import cv2

# Definition of constants
IMAGES_DIR_PATH = '../images'
CLASS_LABEL = 'victory'
DATASET_SIZE = 100

# Create destination directory for images
if not os.path.exists(IMAGES_DIR_PATH):
    os.makedirs(IMAGES_DIR_PATH)

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print('Unable to read camera feed')
else:
    # Create a directory for the class label
    if not os.path.exists(os.path.join(IMAGES_DIR_PATH, CLASS_LABEL)):
        os.makedirs(os.path.join(IMAGES_DIR_PATH, CLASS_LABEL))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display a message to start collecting images
        cv2.putText(frame, 'Press "Q" to start collecting images for {}'.format(CLASS_LABEL),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Image Collector', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    # Check if already exists collected images for the class label
    previous_images = 0

    if os.path.exists(os.path.join(IMAGES_DIR_PATH, CLASS_LABEL)):
        # Get the number of images for the class label
        previous_images = len(os.listdir(
            os.path.join(IMAGES_DIR_PATH, CLASS_LABEL)))

    # Collect DATASET_SIZE images for the class label
    counter = 0
    while counter < DATASET_SIZE:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display frame
        cv2.imshow('Image Collector', frame)

        # Press Q on keyboard to  exit
        cv2.waitKey(25)

        # Save image
        cv2.imwrite(os.path.join(IMAGES_DIR_PATH, CLASS_LABEL,
                    '{}.jpg'.format(counter + previous_images)), frame)

        # Increment counter
        counter += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
