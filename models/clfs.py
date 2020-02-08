from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def run_sequential(X_train, X_test, y_train, y_test):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = Sequential()
    # model.add(Dense(128, activation='relu', input_dim=128))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(150, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(9, activation='relu'))
    # model.add(Dense(5, activation='softmax'))
    model.add(Dense(100, activation='relu', input_dim=128))
    # model.add(Dense(50, activation='relu'))

    # model.add(Dense(80, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0)
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    # print('Accuracy on test data: {}'.format(acc))
    return acc


def run_logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=200)

    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


def run_svm(X_train, X_test, y_train, y_test):
    clf = svm.SVC(gamma='scale')

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)


def run_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def run_knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def run_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=0, n_estimators=100)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def run_mlp(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=0, hidden_layer_sizes=(100,), solver='sgd', max_iter=700)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
