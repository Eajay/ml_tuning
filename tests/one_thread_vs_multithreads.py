from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
import sys
sys.path.append('../')
from ml_tuning import kNN_tuning

def load_data():
    mnist = load_digits()
    data = mnist.data
    label = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=6)
    return X_train, X_test, y_train, y_test


def one_thread():
    knn = kNN_tuning.kNN_tuning()
    X_train, X_test, y_train, y_test = load_data()
    knn.fit(X_train, y_train)
    start = time.time()
    neighbor_list = [2, 3, 5, 6, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600]
    res = knn.predict(X_test, y_test, thread_number=1, neighbor_list=neighbor_list)
    for i, val in enumerate(neighbor_list):
        print(val, " accuracy: ", res[i].accuracy)
    end = time.time()
    print("1 thread time consuming: ", end - start, "s")
    print("*" * 30)


def multi_threads():
    knn = kNN_tuning.kNN_tuning()
    X_train, X_test, y_train, y_test = load_data()
    knn.fit(X_train, y_train)
    start = time.time()
    neighbor_list = [2, 3, 5, 6, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600]
    res = knn.predict(X_test, y_test, thread_number=6, neighbor_list=neighbor_list)
    for i, val in enumerate(neighbor_list):
        print(val, " accuracy: ", res[i].accuracy)
    end = time.time()
    print("6 threads time consuming: ", end - start, "s")


if __name__ == '__main__':
    one_thread()
    multi_threads()





