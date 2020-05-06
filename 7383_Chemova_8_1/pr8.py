from datetime import datetime
import numpy as np
from keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

from var5 import gen_data

class my_callback(Callback):

    def __init__(self, val, prefix='my_model_number', key='val_loss', date=datetime.now()):
        self.val = val
        self.prefix='{}_{}_{}_{}_'.format(date.day, date.month, date.year, prefix)
        self.loss = {}
        self.key = key
        self.index = 0

    def on_train_begin(self, logs=None):
        loss = self.model.evaluate(self.val[0], self.val[1])[0]
        for i in range(1,4):
            self.loss[self.prefix + str(i)] = loss
        for key in self.loss.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs=None):
        for i in range(1, 4):
            if logs.get(self.key) < self.loss[self.prefix + str(i)] and i > self.index:
                self.loss[self.prefix + str(i)] = logs.get(self.key)
                self.model.save(self.prefix + str(i))
                self.index += 1
                break
            elif i <= self.index:
                continue
        if (self.index == 3):
            self.index = 0

    def on_train_end(self, logs=None):
        for (key, loss) in self.loss.items():
            print(key + ' ' + str(loss))

size = 3000
test_count = size // 5
data, labels = gen_data(size)
labels = np.asarray(labels).flatten()

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

temp = list(zip(data, labels))
np.random.shuffle(temp)
data, labels = zip(*temp)
data = np.asarray(data).reshape(size, 50, 50, 1)
labels = np.asarray(labels).flatten()

test_data = data[:test_count]
test_labels = labels[:test_count]
train_data = data[test_count:size]
train_labels = labels[test_count:size]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=12, batch_size=128, validation_data=(test_data, test_labels),
                    callbacks=[my_callback((test_data, test_labels))])
model.evaluate(test_data, test_labels)
