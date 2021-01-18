import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
data = load_iris()

x = data.data
y = data.target_names[data.target]

print(x[:5])
print(y[:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)
predictions = knn_classifier.predict(x_test)
print(predictions)

from sklearn import metrics

knntrainpredict = knn_classifier.predict(x_train)
knntestpredict = knn_classifier.predict(x_test)

print("Training accuracy Score is : ", metrics.accuracy_score(y_train, knntrainpredict))
print("Testing accuracy Score is : ", metrics.accuracy_score(y_test, knntestpredict))
print("Training Confusion Matrix is : \n", metrics.confusion_matrix(y_train, knntrainpredict))
print("Testing Confusion Matrix is : \n", metrics.confusion_matrix(y_test, knntestpredict))
