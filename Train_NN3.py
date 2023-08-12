import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import product
import statsmodels.api as sm

from DataPipeline import DataPipeline, MSE, R2, r2_metrics

file_path = '/Users/zequnli/LocalData/'

# load data
dataset = DataPipeline(file_path)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, layers, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Ones
import keras_tuner as kt

def model_builder(hp):
    input_tensor = Input(shape=(79,))
    # Tune l1 regularization
    # Choose an optimal value between 0.001, 0.0001, or 0.00001
    hp_l1 = hp.Choice('l1', values=[1e-3, 1e-4, 1e-5])
    # Tune learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    # Tune the dropout rate for better regularization
    # Choose an optimal value from 0.0, 0.3, or 0.5
    hp_dropout = hp.Choice('dropout', values=[0.0, 0.3, 0.5])
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 4-32
    hp_units = hp.Int('units', min_value = 8, max_value = 32, step = 8)

    x = layers.Dense(units=32, activation='relu', activity_regularizer = regularizers.L1(hp_l1))(input_tensor)
    x = layers.Dropout(hp_dropout)(x)
    x = layers.Dense(units=16, activation='relu', activity_regularizer = regularizers.L1(hp_l1))(x)
    x = layers.Dense(units=8, activation='relu')(x)
    output_tensor = layers.Dense(1)(x)
    model = Model(input_tensor, output_tensor)


    # set early stopping monitor so the model stops training when it won't improve anymore
    
    optimizer = Adam(learning_rate=hp_lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=r2_metrics)

    return model




for year in range(1987,2022):
    X_train, y_train, X_test, y_test = dataset.LoadTrainTest(year-12, 12)
    tuner = kt.Hyperband(model_builder, objective = 'val_loss', max_epochs=100, factor=3, directory='my_dir', project_name='kt'+str(year))
    early_stopping_monitor = EarlyStopping(monitor='val_r2_metrics', patience=3)
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping_monitor])
    best_hps = tuner.get_best_hyperparameters()[0]

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    model = tuner.hypermodel.build(best_hps)
    model.fit(X, y, epochs=100, validation_split=0.2, callbacks=[early_stopping_monitor])

    #save model
    model.save('NN3_'+str(year)+'.keras')