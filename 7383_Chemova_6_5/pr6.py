import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

from var5 import gen_data

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
history = model.fit(train_data, train_labels, epochs=12, batch_size=128, validation_data=(test_data, test_labels))
model.evaluate(test_data, test_labels)
