import sklearn
import tensorflow
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

print(x_train, y_train)  # after the run 0 represent malignant and 1 represent benign.

classes = ['malignant', 'benign']

clf = svm.SVC(kernel = "linear", C = 1)  # SVC = Support Vector Classification
# The SVM algorithms, function of kernel is to take data as input and transform it into the required form.
# functions examples -  linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.
# kernel helps in increasing accuracy of model.
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)


