import numpy as np


def array_split():
    a = np.array([1, 2, 3, 4, 5, 6, 7])
    offset = int(a.shape[0] * 0.8)
    print(a[:offset])
    print(a[offset:])


def array_delete(array):
    temp = np.delete(array, 3-1, axis=0)
    for i in range(len(temp)):
        print(i)
        print(temp[i])


if __name__ == '__main__':
    a = [[1, 1, 1, 1], [2, 2], [3, 3, 3], [4]]
    array_delete(a)
