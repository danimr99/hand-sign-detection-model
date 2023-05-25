import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skl2onnx import convert_sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import get_latest_tested_opset_version
from onnxmltools.utils import save_model

# Definition of constants
DATASET_FILE_PATH = '../../dataset/dataset.pickle'
MODEL_FILE_PATH = '../../models/scikit/model'

# Load data
data_dict = pickle.load(open(DATASET_FILE_PATH, 'rb'))

# Get data and labels
data = np.asarray([np.asarray(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print('Accuracy: {}'.format(score * 100))

# Check if exists model file path directory
if not os.path.exists(os.path.dirname(MODEL_FILE_PATH)):
    os.makedirs(os.path.dirname(MODEL_FILE_PATH))

# Save model as pickle file
f = open('{}.pickle'.format(MODEL_FILE_PATH), 'wb')
pickle.dump({'model': model}, f)
f.close()

# Save model as onnx file
target_opset = get_latest_tested_opset_version()
n_features = x_train.shape[1]
onnx_model = convert_sklearn(
    model,
    'RandomForestClassifier',
    initial_types=[('input', FloatTensorType([None, n_features]))],
    target_opset={"": target_opset, "ai.onnx.ml": 1}
)
save_model(onnx_model, '{}.onnx'.format(MODEL_FILE_PATH))
