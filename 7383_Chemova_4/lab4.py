import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers

def load_img(path):
    img = load_img(path=path, target_size=(28, 28))
    return img_to_array(img)

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()
print(train_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256 , activation='relu'))
model.add(Dense(10, activation='softmax'))

def optimizer_research(optimizer):
    optimizerConf = optimizer.get_config()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_images, train_labels, epochs=4, batch_size=128, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('optimizer', optimizerConf)
    plt.title('Training and test accuracy')
    plt.plot(history.history['acc'], 'r', label='train')
    plt.plot(history.history['val_acc'], 'b', label='test')
    plt.legend()
    plt.savefig("%s_%s_%s_acc.png" % (optimizerConf["name"], optimizerConf["learning_rate"], test_acc), format='png')
    plt.clf()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.savefig("%s_%s_%s_loss.png" % (optimizerConf["name"], optimizerConf["learning_rate"], test_acc), format='png')
    plt.legend()
    plt.clf()
    return model

for learning_rate in [0.001, 0.01]:
    optimizer_research(optimizers.Adam(learning_rate=learning_rate))
    optimizer_research(optimizers.RMSprop(learning_rate=learning_rate))
    optimizer_research(optimizers.Adagrad(learning_rate=learning_rate))
    optimizer_research(optimizers.SGD(learning_rate=learning_rate))
