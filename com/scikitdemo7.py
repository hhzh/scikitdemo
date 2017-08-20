from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

import pickle

# with open('save/clf.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# with open('save/clf.pickle', 'rb') as f:
#     clf2=pickle.load(f)
#     print(clf2.predict(X[0:1]))

from sklearn.externals import joblib
joblib.dump(clf,'save.pkl')

clf3=joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))