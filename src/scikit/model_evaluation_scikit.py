import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev

# Definition of constants
DATASET_FILE_PATH = '../../dataset/dataset.pickle'
STRATIFIED_KFOLD_SPLITS = 5

# Load data
data_dict = pickle.load(open(DATASET_FILE_PATH, 'rb'))

# Get data and labels
data = np.asarray([np.asarray(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Create a random forest classifier
model = RandomForestClassifier()

# Split data into training and testing sets using stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=STRATIFIED_KFOLD_SPLITS, shuffle=True)

# Train the model and evaluate accuracy for each fold
stratified_accuracy = []

for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
    x_train_fold, x_test_fold = data[train_index], data[test_index]
    y_train_fold, y_test_fold = labels[train_index], labels[test_index]
    model.fit(x_train_fold, y_train_fold)
    stratified_accuracy.append(model.score(x_test_fold, y_test_fold))

# Print the output.
print('List of possible accuracy:', stratified_accuracy)
print('Maximum Accuracy:',
      max(stratified_accuracy)*100, '%')
print('Minimum Accuracy:',
      min(stratified_accuracy)*100, '%')
print('Overall Accuracy:',
      mean(stratified_accuracy)*100, '%')
print('Standard Deviation:', stdev(stratified_accuracy))