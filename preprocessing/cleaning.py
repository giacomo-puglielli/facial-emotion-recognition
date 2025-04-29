import numpy as np
import keras
from keras.utils import to_categorical

def preprocess_x(df):
    x_t = []
    for index, row in df.iterrows():
        k = row['pixels'].split(" ")
        x_t.append(np.array(k))
    x_t = np.array(x_t, dtype = 'uint8')
    x_t = x_t.reshape(x_t.shape[0], 48, 48, 1)
    return x_t

def preprocess_y(df):
    y_t = []
    for index, row in df.iterrows():
        y_t.append(row['emotion'])
    y_t = np.array(y_t, dtype = 'uint8')
    y_t = to_categorical(y_t, num_classes=7)
    return y_t
    
        
