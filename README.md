# Hand Sign Detection

This repository contains the implementation of a machine learning model for hand sign detection using Mediapipe Hands with the following libraries:
	- SciKit Learn
	- TensorFlow Decision Forests

## USAGE

1. Install dependencies

	**NOTE**: You can clone the environment from [environment.yml](https://github.com/danimr99/hand-sign-detection/blob/main/environment.yml) using Conda.
	
2. Use the scripts in the following order to collect images to create your dataset:
	## Image Collector
	Collects images using the webcam for a class label. If already exists a folder for the class label specified on the destination images directory, it will add more images to it.

	#### Instructions
	It pop-ups a webcam window that will save a specified number of images for a class label. It will start collecting images on press the "Q" key button.

	#### Modifiable Constants
	- IMAGES_DIR_PATH: Directory or path to save the collected images.
	- CLASS_LABEL: String for the class label to collect.
	- DATASET_SIZE: Number of images to collect.



	## Dataset Generator
	Processes all the images collected using Mediapipe Hands in order to extract the normalized coordinates of the landmarks of each hand. It generates a file with all the dataset data.

	#### Modifiable Constants
	- IMAGES_DIR_PATH: Directory or path to get the collected images from.
	- EXCLUDED_FILES: List of excluded file names.
	- OUTPUT_FILE_PATH: File name or path to save the dataset data.

3. Depending on the library used to implement the model, use the scripts in the following order to train and test a live demo with your own dataset.

	## Using SciKit Learn
	
	### Train Classifier SciKit
	Trains a RandomForestClassifier model based on the dataset data file generated by the previous script/step. This script generates both a .pickle file and a .onnx file to export the model.

	##### Modifiable Constants
	- DATASET_FILE_PATH: File name or path to get the dataset data from.
	- MODEL_FILE_PATH: File name or path to save the model (without extension).

	##### Converting model for Android/iOS
	Although it could be directly used, the generated .onnx file must be converted to .ort for a reduced size build. To do so, you must execute the following command from the terminal:
	
	```console
		$ python3 -m onnxruntime.tools.convert_onnx_models_to_ort ONNX_FILE_NAME
	```



	### Live Detection SciKit
	Detects hand gestures based on the model file generated by the previous script/step. This script uses a .pickle file to load the model.

	##### Modifiable Constants 
	- MODEL_FILE_PATH: File name or path to get the model from.
	- MIN_PREDICTION_CONFIDENCE: From 0 to 1, the minimum prediction confidence to consider a prediction as valid.
	- CONSECUTIVE_PREDICTIONS_FRAMES: Number of consecutive frames with the same prediction to consider a detection as valid.
	- LABELS_FILE_PATH: File name or path to save the model labels in a JSON format.



	## Using TensorFlow Decision Forests

	### Train Classifier Tensorflow
	Trains a TensorFlow Decision Forest model based on the dataset data file generated by the previous script/step. This script generates a Tensorflow model both in SavedModel and TFJS format.

	##### Modifiable Constants
	- DATASET_FILE_PATH: File name or path to get the dataset data from.
	- TF_MODEL_FILE_PATH: File name or path to save the TensorFlow model in SavedModel format (without extension).
	- TFJS_MODEL_FILE_PATH: File name or path to save the TensorFlowJS model (without extension).
	- LABELS_FILE_PATH: File name or path to save the model labels in a JSON format.



	### Live Detection Tensorflow
	Detects hand gestures based on the Tensorflow model generated by the previous script/step. This script uses a Tensorflow model.

	##### Modifiable Constants 
	- DATASET_FILE_PATH: File name or path to get the dataset file.
	- MODEL_FILE_PATH: File name or path to get the model from.
	- MIN_PREDICTION_CONFIDENCE: From 0 to 1, the minimum prediction confidence to consider a prediction as valid.
	- CONSECUTIVE_PREDICTIONS_FRAMES: Number of consecutive frames with the same prediction to consider a detection as valid.