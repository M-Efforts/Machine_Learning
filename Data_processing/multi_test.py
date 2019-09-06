from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# this will generate a random multi-label dataset
X, y = make_multilabel_classification(sparse=True, n_labels=20, return_indicator='sparse', allow_unlabeled=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# for i in X_train[15]:
#     print(i)
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
