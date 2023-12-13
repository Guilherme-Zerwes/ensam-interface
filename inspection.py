import numpy as np
import pandas as pd

def verify_str(data):
    return False if len(['Nok' for types in data.dtypes if types not in ['float64', 'int64']]) >= 1 else True

def very_NaN(data):
    return not data.isnull().values.any()

def verify_targ(data):
    cond1 = False if data.dtypes not in ['int64', 'object'] else True
    cond2 = len(data.unique())/len(data) < 0.8
    return cond2 and cond1