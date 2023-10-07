import math
import os
import random
import sys

import numpy as np

"""
var range must be a n*2 array
eacl row must corresponds to lower and upper limit of each variable
"""


def GenerateLatinHyperCube(num, var_range):
    coord = []
    for range_ in var_range:
        # lower upper limit so the latin hyper cube won't goes into upper_limit + step size
        range_[1] = (range_[1] + range_[0]/num)/(1 + 1./num)
        lower_limit, step_size = np.linspace(
            range_[0], range_[1], num=num, retstep=True
        )
        upper_limit = np.linspace(
            range_[0] + step_size,
            range_[1] + step_size,
            num=num)

        coord.append(
            np.random.uniform(
                low=lower_limit,
                high=upper_limit,
                size=num))

    coord = np.array(coord)
    # shuffle all dimensions for good measure
    for index in range(0, coord.shape[0]):
        np.random.shuffle(coord[index])

    return coord.T


def GenerateRandomLattice(num, var_range):
    coord = []
    for index in range(0, num):
        coord.append([random.uniform(range_[0], range_[1])
                      for range_ in var_range])

    return np.array(coord)


def GenerateRegularLattice(num, var_range):
    """
    num here must be a 3 element which corresponds to number of element in each dimension
    """
    var_num = float(len(var_range))
    num = int(num ** (1.0 / var_num) + 0.5)
    coord = []
    for index, range_ in enumerate(var_range):
        coord.append(
            np.repeat(
                np.linspace(range_[0], range_[1], endpoint=True, num=num).tolist()
                * (num ** index),
                num ** (var_num - index - 1),
            ).tolist()
        )

    return np.array(coord).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    content = GenerateRegularLattice(125, [[-3, 3], [-3, 3], [0, 3]])
    print(content)
    df = pd.DataFrame(
        content, columns=["col%d" % num for num in range(0, content.shape[1])]
    )
    pd.tools.plotting.scatter_matrix(df, alpha=1)
    plt.show()

    content = GenerateLatinHyperCube(100, [[-3, 3], [-3, 3], [0, 3]])
    df = pd.DataFrame(
        content, columns=["col%d" % num for num in range(0, content.shape[1])]
    )
    pd.tools.plotting.scatter_matrix(df, alpha=1)
    plt.show()

    content = GenerateRandomLattice(100, [[-3, 3], [-3, 3], [0, 3]])
    df = pd.DataFrame(
        content, columns=["col%d" % num for num in range(0, content.shape[1])]
    )
    pd.tools.plotting.scatter_matrix(df, alpha=1)
    plt.show()
