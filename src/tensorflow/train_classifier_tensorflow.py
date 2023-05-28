import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflowjs as tfjs

# Definition of constants
DATASET_FILE_PATH = '../../dataset/dataset.pickle'
TF_MODEL_FILE_PATH = '../../models/tensorflow/tf_model'
TFJS_MODEL_FILE_PATH = '../../models/tensorflow/tfjs_model'

# Load data
data_dict = pickle.load(open(DATASET_FILE_PATH, 'rb'))

# Get data and labels
data = np.asarray([np.asarray(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Convert each label into an integer identifier depending on the label (same label = same identifier)
unique_labels = np.unique(labels)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
labels = np.asarray([label_to_id[label] for label in labels])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert pandas DataFrame to TensorFlow Dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)

# Create a Random Forest Model
model = tfdf.keras.RandomForestModel()

# Train the model
model.fit(train_ds)

# Compile the model
model.compile(metrics=['accuracy'])

# Evaluate the model
evaluation = model.evaluate(test_ds, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Check if exists TensorFlow model file path directory
if not os.path.exists(os.path.dirname(TF_MODEL_FILE_PATH)):
    os.makedirs(os.path.dirname(TF_MODEL_FILE_PATH))

# Save Tensorflow model
model.save(TF_MODEL_FILE_PATH)

# Load the model with Keras
model = tf.keras.models.load_model(TF_MODEL_FILE_PATH)

# Check if exists TensorFlowJS model file path directory
if not os.path.exists(os.path.dirname(TFJS_MODEL_FILE_PATH)):
    os.makedirs(os.path.dirname(TFJS_MODEL_FILE_PATH))

# Convert the keras model to TensorFlow.js
tfjs.converters.tf_saved_model_conversion_v2.convert_keras_model_to_graph_model(
    model, TFJS_MODEL_FILE_PATH)
