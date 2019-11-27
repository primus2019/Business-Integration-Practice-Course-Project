import numpy as np 
import pandas as pd 
import math 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


def cut(x:np.ndarray, y:np.ndarray, **kwargs)->list:
    threshold = None
    kwargs.setdefault("max_depth", 3)
    kwargs.setdefault("min_samples_leaf", max(int(x.size * 0.1), 1))
    tree = DecisionTreeClassifier(**kwargs)
    tree.fit(x.reshape(-1, 1), y)
    threshold = Counter(tree.tree_.threshold)
    temp = threshold.most_common(1)[0][0]
    threshold.pop(temp)
    threshold = sorted(list(threshold.keys()))
    if threshold[0] <= x.min():
        threshold[0] = -math.inf
    else:
        threshold.insert(0, -math.inf)
    if threshold[-1] >= x.max():
        threshold[-1] = math.inf
    else:
        threshold.append(math.inf)
    return threshold
