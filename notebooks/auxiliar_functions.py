'''
Auxiliar functions
'''

import pandas as pd
import itertools
from collections import defaultdict
from itertools import permutations,combinations
import numpy as np
import matplotlib.pylab as plt
from functools import reduce

import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score,mean_squared_error, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

import datetime, itertools, math

from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from itertools import product
from tqdm import tqdm_notebook


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')


from xgboost import XGBRegressor
from sklearn.metrics import median_absolute_error


def class_to_name(class_name):
    return(str(class_name).split('(')[0])

def apply_xgb(df, cols_to_drop, y, features_new):

    features_new, y = df.iloc[:,:-1], df.iloc[:,-1]
    features_new = features_new.drop(cols_to_drop, axis=1)
    X_new, y_new = features_new.as_matrix(),  y.as_matrix()

    my_model = XGBRegressor()
    my_model.fit(X_new, y_new, verbose=False)

    predictions = my_model.predict(X_new)

    df_res = lists_to_df(predictions, y_new, 'predictions','true values')

    mae = round(median_absolute_error(list(df_res['predictions']), list(df_res['true values'])),3)

    print('Median Absolute Error:', mae)

    return df_res, my_model


def apply_linregRidge(df, cols_to_drop, y, features_new):

    features_new, y = df.iloc[:,:-1], df.iloc[:,-1]
    features_new = features_new.drop(cols_to_drop, axis=1)
    X_new, y_new = features_new.as_matrix(),  y.as_matrix()

    my_model = Ridge()
    my_model.fit(X_new, y_new)

    predictions = my_model.predict(X_new)

    df_res = lists_to_df(predictions, y_new, 'predictions','true values')

    mae = round(median_absolute_error(list(df_res['predictions']), list(df_res['true values'])),3)

    print('Median Absolute Error:', mae)

    return df_res, my_model


def plot_function(fig_size, x, y1, y2, label_1, label_2, style_1, style_2):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(x, y1, 'r-', label=label_1, linestyle=style_1)
    ax.plot(x, y2,
            'b-', label=label_2, linestyle=style_2)
    ax.legend(loc='best');

def apply_linregLasso(df, cols_to_drop, y, features_new):

    features_new, y = df.iloc[:,:-1], df.iloc[:,-1]
    features_new = features_new.drop(cols_to_drop, axis=1)
    X_new, y_new = features_new.as_matrix(),  y.as_matrix()

    my_model = Lasso()
    my_model.fit(X_new, y_new)

    predictions = my_model.predict(X_new)

    df_res = lists_to_df(predictions, y_new, 'predictions','true values')

    mae = round(median_absolute_error(list(df_res['predictions']), list(df_res['true values'])),3)

    print('Median Absolute Error:', mae)

    return df_res, my_model

def apply_linregElastic(df, cols_to_drop, y, features_new):

    features_new, y = df.iloc[:,:-1], df.iloc[:,-1]
    features_new = features_new.drop(cols_to_drop, axis=1)
    X_new, y_new = features_new.as_matrix(),  y.as_matrix()

    my_model = ElasticNet()
    my_model.fit(X_new, y_new)

    predictions = my_model.predict(X_new)

    df_res = lists_to_df(predictions, y_new, 'predictions','true values')

    mae = round(median_absolute_error(list(df_res['predictions']), list(df_res['true values'])),3)

    print('Median Absolute Error:', mae)

    return df_res, my_model

def apply_linreg(df, cols_to_drop, y, features_new):

    features_new, y = df.iloc[:,:-1], df.iloc[:,-1]
    features_new = features_new.drop(cols_to_drop, axis=1)
    X_new, y_new = features_new.as_matrix(),  y.as_matrix()

    my_model = LinearRegression()
    my_model.fit(X_new, y_new)

    predictions = my_model.predict(X_new)

    df_res = lists_to_df(predictions, y_new, 'predictions','true values')

    mae = round(median_absolute_error(list(df_res['predictions']), list(df_res['true values'])),3)

    print('Median Absolute Error:', mae)

    return df_res, my_model


def plot_func(s1, s2, xlabel, ylabel, ymax, ymin, xmax, xmin, dim1, rotation):

    plt.rcParams["figure.figsize"] = dim1
    x_values = np.arange(min(s1),max(s1))
    plt.plot(s1,'black');
    plt.plot(s2, 'b');
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.ylim(ymax)
    plt.ylim(ymin)
    plt.xlim(xmax)
    plt.xlim(xmin)
    plt.xticks(rotation=rotation);
    plt.tick_params(labelsize=12)
    plt.show();

    return

def plot_func_new(s1, s2, xlabel, ylabel, ymax, ymin, xmax, xmin, dim1, rotation, x_values):

    plt.rcParams["figure.figsize"] = dim1
    x_values = x_values
    plt.plot(s1,'black');
    plt.plot(s2, 'b');
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.ylim(ymax)
    plt.ylim(ymin)
    plt.xlim(xmax)
    plt.xlim(xmin)
    plt.xticks(rotation=rotation);
    plt.tick_params(labelsize=12)
    plt.show();

    return

def power_set(lst):
    ps = [list(j) for i in range(len(lst))
          for j in itertools.combinations(lst, i+1)]
    return ps

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def coalitions(s):
    l = len(s)
    if l==1:
        return s
    else:
        channels=[]
        for i in range(1,l+1):
            channels.extend(map(list,itertools.combinations(s, i)))
    return list(map(",".join,map(sorted,channels)))

def v(C,d):
    sub = coalitions(C.split(","))
    worth=0
    for s in sub:
        if s in d:
            worth += d[s]
    return worth

def lists_to_df(lst1,lst2,col_1,col_2):
    return pd.DataFrame({col_1: lst1,col_2: lst2})


def s_to_df(s, idx, col_name):
    s = s.to_frame().reset_index()
    s.columns = [idx, col_name]
    return s

def lists_to_df(lst1,lst2,col_1,col_2):
    return pd.DataFrame({col_1: lst1,col_2: lst2})


def multi_replace(t, d):
    for i, j in d.items():
        t = t.replace(i, j)
        t = t.lower()
    return t

def unique_touch_points(row):
    lst = sorted(row.split(','))
    return ','.join(set(lst))
