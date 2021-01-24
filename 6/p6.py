import pandas as pd

msg = pd.read_csv('6\pg6.csv',names=['message','label'])
print('The dimensions of the dataset', msg.shape)

msg['labelnum'] = msg.label.map({'pos':1,'neg':0})
x = msg.message
y = msg.labelnum

#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y)

#output of count vectoriser is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm  = count_vect.transform(xtest)

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

#printing accuracy metrics
from sklearn import metrics

print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))