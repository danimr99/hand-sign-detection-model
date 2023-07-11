import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix, classification_report, plot_confusion_matrix
from statistics import mean, stdev

# Definition of constants
DATASET_FILE_PATH = '../../dataset/dataset.pickle'
STRATIFIED_KFOLD_SPLITS = 5

# Load data
data_dict = pickle.load(open(DATASET_FILE_PATH, 'rb'))

# Get data and labels
data = np.asarray([np.asarray(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])
unique_labels = np.unique(labels)

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

# Print the results for k-fold cross-validation
print('Stratified k-fold cross-validation')
print('List of possible accuracies:', stratified_accuracy)
print('Maximum Accuracy:',
      max(stratified_accuracy)*100, '%')
print('Minimum Accuracy:',
      min(stratified_accuracy)*100, '%')
print('Overall Accuracy:',
      mean(stratified_accuracy)*100, '%')
print('Standard Deviation:', stdev(stratified_accuracy))
print('\n')

# Train the model and evaluate accuracy for the whole dataset
model.fit(data, labels)
print('Accuracy of the whole dataset:', model.score(data, labels)*100, '%')
print('\n')

# Confusion matrix
y_pred = model.predict(data)
confusion_matrix = multilabel_confusion_matrix(labels, y_pred, labels=np.unique(labels))
report = classification_report(labels, y_pred)

# Iterate over the confusion matrixes
for i, cm in enumerate(confusion_matrix):
      print('Confusion matrix for label', unique_labels[i])

      # Extract true negatives, false positives, false negatives and true positives
      tn, fp, fn, tp = cm.ravel()

      # Print each value
      print('True negatives:', tn)
      print('False positives:', fp)
      print('False negatives:', fn)
      print('True positives:', tp)
      print('\n')

print('\n')
print('Classification report')
print(report)
print('\n')

# Plot the confusion matrix
color = 'white'
matrix = plot_confusion_matrix(model, data, y_pred, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()