from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression

model = Doc2Vec.load('./twitter.d2v')

train_arrays = numpy.zeros((750000, 100))
train_labels = numpy.zeros(750000)

for i in range(375000):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[375000 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[375000 + i] = 0


test_arrays = numpy.zeros((750000, 100))
test_labels = numpy.zeros(750000)

for i in range(375000):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[375000 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[375000 + i] = 0

# for i in range(50000):
#     prefix_unsup = 'TEST_POS_' + str(i)
#     test_arrays[i] = model.docvecs[prefix_test_pos]
#     test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
#     test_labels[i] = 1
#     test_labels[12500 + i] = 0


classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)


score = classifier.score(test_arrays, test_labels)

print(score)
