# %%
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed

# %%
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2].resolve()))
from Data import preprocess as data_generator

# %%
seed(911)


# %%
# same metrics to run after models
def run_metrics(model, X_test, y_test):
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    print(f"Accuracy to beat: {y_pred.value_counts()[1]/y_pred.value_counts().sum()}")

    # assumes .5 as the round value
    y_pred_class = model.predict_classes(X_test, verbose=1)

    # get number of epochs ran for
    epoch = len(model.history.history['loss'])

    print('=========================')
    print(f'Confusion Matrix {epoch} epoch')
    print('=========================')
    cm = confusion_matrix(y_test, y_pred_class)
    print('True negatives: ',  cm[0,0])
    print('False negatives: ', cm[1,0])
    print('False positives: ', cm[0,1])
    print('True positives: ',  cm[1,1]) 

    print(cm)

    # precision = tp / tp + fp
    print(f"Precision: {cm[1,1] / (cm[1,1] + cm[0,1])}")
    # recall = tp / tp + fn
    print(f"Recall: {cm[1,1] / (cm[1,1] + cm[1,0])}")

    # precision = tn / tn + fn
    print(f"Neg Precision: {cm[0,0] / (cm[0,0] + cm[1,0])}")
    # recall = tn / tn + fp
    print(f"Neg Recall: {cm[0,0] / (cm[0,0] + cm[0,1])}")

# %%
# save sequential model layers to config for reuse

def createmodel(coleng):
    model = keras.Sequential()
    model.add(Dense(16, input_dim=len(coleng.columns), activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(1))
    model_config = model.get_config()
    return model_config



### MODEL TESTING ###


# %%
# build a standard model

X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=True,
    balance='undersample',
    scale=True,
    shuffle=True, 
    drop_insig=True)

model_config = createmodel(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


standard_model = keras.Sequential.from_config(model_config)
standard_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
standard_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=0,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(standard_model, X_pred, y_pred)


# %%
# build a model with drop 202020 semester
X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    drop_202020=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=True,
    balance='undersample',
    scale=True,
    shuffle=True)

model_config = createmodel(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

drop202020_model = keras.Sequential.from_config(model_config)
drop202020_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
drop202020_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=0,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(drop202020_model, X_pred, y_pred)

# %%
# build a model with minmax scale
X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    drop_202020=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=False,
    balance='undersample',
    scale='minmax',
    shuffle=True)
X = X[features]
X_pred = X_pred[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

minmax_model = keras.Sequential.from_config(model_config)
minmax_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
minmax_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=0,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(minmax_model, X_pred, y_pred)

# %%
# build a model with PoG flipped
X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=False,
    balance='undersample',
    scale=True,
    shuffle=True,
    flip_pog=True)
X = X[features]
X_pred = X_pred[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

flippog_model = keras.Sequential.from_config(model_config)
flippog_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
flippog_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=0,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(flippog_model, X_pred, y_pred)

# %%
# just upperclassmen
X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=True,
    balance='undersample',
    scale=True,
    shuffle=True,
    subset='upperclass')

model_config = createmodel(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

upperclass_model = keras.Sequential.from_config(model_config)
upperclass_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
upperclass_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=64, 
    verbose=1,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(upperclass_model, X_pred, y_pred)

# %%
# is stratify working
_, y, _, _ = data_generator.retention()
y_train, y_test = train_test_split(y, test_size=0.3, stratify=y)

print(y_train.value_counts())
print(y_test.value_counts())

# %%
# test with SMOTE

# just upperclassmen
X, y, X_pred, y_pred = data_generator.retention(
    drop_id=True, 
    drop_summer=True,
    drop_insig=True,
    fill_na=True,
    encode_label=True,
    encode_ohe=True,
    balance='SMOTE',
    scale=True,
    shuffle=True,
    subset='upperclass')

model_config = createmodel(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

upperclass_model = keras.Sequential.from_config(model_config)
upperclass_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
upperclass_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=1,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

run_metrics(upperclass_model, X_pred, y_pred)

# %%
