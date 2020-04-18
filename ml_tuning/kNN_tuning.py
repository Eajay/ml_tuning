from .exceptions import InvalidArgumentType
from .exceptions import InvalidArgumentValue
import threading
import multiprocessing
import numpy as np
from collections import defaultdict

class Result:
    def __init__(self, predict_y, real_y):
        self.predict_y = np.array(predict_y)
        self.real_y = np.array(real_y)
        self.res_dict = defaultdict(lambda: defaultdict(int))
        self.right_prediction = 0

    def set_result_dict(self):
        """

        get the result dictionary for further use, such as precision, recall, F-1 score
        """
        for real, pre in zip(self.real_y, self.predict_y):
            self.res_dict[real][pre] += 1
            if real == pre:
                self.right_prediction += 1

    @property
    def accurate_number(self):
        return self.right_prediction

    @property
    def accuracy(self):
        return self.right_prediction / self.real_y.shape[0]


class kNN_tuning():
    """
    Handling multiple input, get the results of different K values
    Calculate the nearest matrix by multi-threads
    """
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.nearest_nums = None
        self.neighbor_list = None
        self.predict_all = None

    def _set_neighbor_list(self):
        total = self.train_y.shape[0]
        mid = int(np.sqrt(total))
        gap = int(mid / 10)
        self.neighbor_list = [mid - 2*gap, mid - gap, mid, mid + gap, mid + 2*gap]

    def fit(self, train_X=None, train_y=None):
        """
        Check the input types and take them.
        :param train_X: numpy or list
        :param train_y: numpy or list
        :return: No return, only initial basic value
        """
        # check training type
        if not isinstance(train_X, (np.ndarray, list)):
            raise InvalidArgumentType("train_X should be either numpy or list")
        if not isinstance(train_y, (np.ndarray, list)):
            raise InvalidArgumentType("train_y should be either numpy or list")

        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)

    def _thread_calculate_neighbor(self, rows_begin, rows_end):
        """
        thread function, set corresponding nearest neighbor matrix row
        :param rows_begin: input start index
        :param rows_end: input end index
        """
        for row in range(rows_begin, rows_end):
            dist = np.sum(np.square(self.test_X[row] - self.train_X), axis=1)
            self.nearest_nums[row] = np.argsort(dist)

    def _thread_predict_result(self, index, k):
        predict_list = []
        for i in range(self.nearest_nums.shape[0]):
            count = defaultdict(int)
            predict = 0
            for j in range(k):
                tmp = self.train_y[int(self.nearest_nums[i][j])]
                count[tmp] += 1
                if count[tmp] > count[predict]:
                    predict = tmp
            predict_list.append(predict)
        res = Result(predict_list, self.test_y)
        res.set_result_dict()
        self.predict_all[index] = res

    def predict(self, test_X=None, test_y=None, thread_number=1, neighbor_list=None):
        """
        Utilize multi-threads to calculate nearest neighbor matrix,
        for each value in neighbor_list, predict all test cases results.
        :param test_X: numpy or list
        :param test_y: numpy or list
        :param thread_number: an integer range from [0, current_system_maximum_cores]
        :param neighbor_list: User can input a integer list or leave it alone and let function
                              _set_neighbor_list() generate for user.
        :return: result list, each value is Result instance.
        """
        if not isinstance(test_X, (np.ndarray, list)):
            raise InvalidArgumentType("test_X should be either numpy or list")
        if not isinstance(test_y, (np.ndarray, list)):
            raise InvalidArgumentType("test_y should be either numpy or list")

        self.test_X = np.array(test_X)
        self.test_y = np.array(test_y)
        self.nearest_nums = np.empty([self.test_X.shape[0], self.train_y.shape[0]], dtype=float)

        # check thread_number type and value
        if not isinstance(thread_number, int):
            raise InvalidArgumentType("thread_number should be integer")
        if thread_number <= 0:
            raise InvalidArgumentValue("thread_number should larger than 0")
        if thread_number > multiprocessing.cpu_count():
            thread_number = multiprocessing.cpu_count()

        if neighbor_list is None:
            self._set_neighbor_list()
        elif not isinstance(neighbor_list, (np.ndarray, list)):
            raise InvalidArgumentType("neighbor_list should be None or numpy or list")
        else:
            self.neighbor_list = neighbor_list

        # start multi thread processing
        thread_list = []
        thread_cover_rows = int(self.test_X.shape[0] / thread_number)

        for i in range(thread_number):
            thread = threading.Thread(target=self._thread_calculate_neighbor,
                                      args=(i * thread_cover_rows,
                                            min((i + 1) * thread_cover_rows, self.test_X.shape[0])))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        # all each k in neighbor_list generate its own Result
        self.predict_all = [None] * len(neighbor_list)
        thread_list = []
        for i, k in enumerate(neighbor_list):
            thread = threading.Thread(target=self._thread_predict_result,
                                      args=(i, k))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        return self.predict_all





















