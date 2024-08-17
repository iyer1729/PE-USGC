import os
from src.dataloader import load_data
from src.train import train_gmm, train_classifier, evaluate_model, save_model
from src.visualize import plot_pca_clusters, plot_confusion_matrix, plot_roc_curve
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_path = 'data/'

# Load data, change step_size to vary FPS of data read
# Include participant type, layperson (s) or professional (w) to account for different landmark points
X, y = load_data(dataset_path, step_size=1, participant_group='s')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train GMM and RandomForest Classifier
gmm = train_gmm(X_train)
clf = train_classifier(X_train, y_train)

# Evaluate model
accuracy, precision, recall, f1, conf_matrix = evaluate_model(clf, X_test, y_test)

# Print accuracy for each task
print(f'Overall Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# Print accuracy for each class
print('Classification Report:')
print(classification_report(y_test, clf.predict(X_test)))

# Plotting
pca = PCA(n_components=3)
X_pca_train = pca.fit_transform(X_train)
plot_pca_clusters(X_pca_train, gmm.predict(X_train))

# Confusion matrix
plot_confusion_matrix(conf_matrix)

# ROC Curve
y_prob = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc)

# Save models
save_model(clf, 'models/task_classifier.pkl')
save_model(gmm, 'models/gmm_model.pkl')
