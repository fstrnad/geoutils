from sklearn.metrics import explained_variance_score
import scipy.stats as st
from sklearn.svm import SVR
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sklearn as sk
import numpy as np
import pandas as pd
import geoutils.utils.time_utils as tu
import geoutils.utils.statistic_utils as sut
from importlib import reload
# %%
# Fit linear model
reload(sut)


def preproccess_data(data_arr,
                     time_lag=0,
                     ):
    if len(data_arr) > 0:
        df = tu.arr_lagged_ts(ts_arr=data_arr, lag=time_lag)
        return df
    else:
        return None


def fit_model(X, Y, method='linear', degree=2):

    if method == 'linear':
        model = LinearRegression().fit(X, Y)
        # print('slope:', model.coef_)
        # print('intercept:', model.intercept_)
    elif method == 'polynomial':
        X = model = LinearRegression(fit_intercept=False).fit(X, Y)
    elif method == 'svr':
        model = SVR(kernel="rbf", degree=degree)
        model.fit(X, Y)

    return model


def evaluate_data(X_data, Y_data, time_lag=0, test=True,
                  auto_corr_lag=0,
                  method='linear', degree=2, ratio_tt=0.9,
                  ):
    if auto_corr_lag > 0:
        time_lag = auto_corr_lag
        data_arr = [Y_data]
        for data in X_data:
            data_arr.append(data)
        X_pd = preproccess_data(data_arr=data_arr,  # has to be provided as array of data
                                time_lag=auto_corr_lag
                                )
        for idx in range(0, auto_corr_lag):
            X_pd = X_pd.drop(f'0_{idx}', axis=1)  # drop the true values
        print(X_pd)
    else:
        X_pd = preproccess_data(data_arr=X_data,  # has to be provided as array of data
                                time_lag=time_lag,
                                )
    Y_pd = preproccess_data(data_arr=[Y_data],
                            time_lag=0)  # Only one target Variable
    if time_lag > 0:
        Y_pd = Y_pd[:-time_lag]

    if test:
        train_X, test_X, train_Y, test_Y = sk.model_selection.train_test_split(
            X_pd,
            Y_pd,
            test_size=ratio_tt,
            random_state=42)
    else:
        train_X = X_pd
        train_Y = Y_pd

    model = fit_model(X=train_X, Y=train_Y,
                      method=method, degree=degree)
    if test:
        score = model.score(test_X, test_Y)
        print(f'Goodness-of-test-fit: {score:.2f}')

    Y_pred = model.predict(X_pd).flatten()

    fit_statistics(Y_pred, Y_pd['0_0'], model=model)

    return model, Y_pred


def predict_data(X_data, Y_data, X_pred, time_lag=0,
                 method='linear', degree=2, ratio_tt=0.9,
                 time_range=None, start_month='Jan', end_month='Dec'):

    model = evaluate_data(X_data=X_data, Y_data=Y_data,
                          time_lag=time_lag,
                          method=method,
                          degree=degree,
                          ratio_tt=ratio_tt,
                          time_range=time_range,
                          start_month=start_month,
                          end_month=end_month)

    X_pred_pp = preproccess_data(X_pred,
                                 method=method,
                                 degree=degree)
    Y_pred = model.predict(X_pred_pp['data'])
    Y_pred = xr.DataArray(data=np.hstack(Y_pred),
                          dims=X_pred.dims,
                          coords=X_pred.coords,
                          name=f'{method}_pred')

    return Y_pred


def fit_statistics(Y_pred, Y_true, model):
    y_mean = np.mean(Y_true)
    # ie. Rsquare = np.sum((Y_pred-y_mean)**2) / np.sum((Y_true - y_mean)**2)
    Rsquare = explained_variance_score(Y_true, Y_pred)
    print(f'Explained Variance: {Rsquare:.2f}')

    r_pr, p = st.pearsonr(Y_pred, Y_true)
    print(f'Pearson R: {r_pr:.2f}')

    return Rsquare
