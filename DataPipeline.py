import pandas as pd
import numpy as np
import tensorflow as tf

class DataPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path+'mldata.csv')

        self.features = self.data.columns.to_list()
        self.features.remove('exret')
        self.features.remove('yyyymm')
        self.features.remove('permno')
        self.features.remove('me')

    def LoadTrainTest(self, test_year, test_period):
        # Load train and test data
        test = self.data[self.data['yyyymm'] < test_year*100]
        test = self.data[(self.data['yyyymm'] >= test_year*100)&(self.data['yyyymm'] < (test_year+test_period)*100)]

        #fill missing values with 0
        test = test.fillna(0)
        test = test.fillna(0)

        # Split train and test data into X and y
        X_train = test[self.features]
        y_train = test['exret']
        X_test = test[self.features]
        y_test = test['exret']

        return X_train, y_train, X_test, y_test
    

    def LoadOneYearXY(self, year):
        one_year = self.data[(self.data['yyyymm'] > year*100)&(self.data['yyyymm'] < (year+1)*100)]
        one_year = one_year.fillna(0)
        X_one_year = one_year[self.features]
        y_one_year = one_year['exret']
        return X_one_year, y_one_year, one_year[['yyyymm','permno','exret','me']]

    def LoadOneMonthXY(self, year, month):
        yyyymm = year*100 + month
        one_month = self.data[self.data['yyyymm'] == yyyymm]
        one_month = one_month.fillna(0)
        X_one_month = one_month[self.features]
        y_one_month = one_month['exret']
        return X_one_month, y_one_month, one_month[['yyyymm','permno','exret','me']]

# Define a function to calculate the MSE and R-squared
def MSE(y_true, y_pred):
    y = np.array(y_true)
    y_hat = np.array(y_pred)
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    return np.mean(np.square(y-y_hat))

def R2(y_true, y_pred):
    y = np.array(y_true)
    y_hat = np.array(y_pred)
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    return 1 - np.sum(np.square(y-y_hat))/np.sum(np.square(y))

# define customized r2_metrics for tensorflow use
def r2_metrics(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.dtypes.cast(y_true, tf.float64)
    y_pred = tf.dtypes.cast(y_pred, tf.float64)
    return 1 - tf.reduce_sum(tf.square(y_true-y_pred))/tf.reduce_sum(tf.square(y_true))


