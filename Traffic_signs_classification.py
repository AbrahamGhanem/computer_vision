import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
root = r'C:\Users\GHANEM\Desktop\OpenCV\data\GTSRB'
'''
for c in range(classes):
    path = os.path.join(root, 'Train', str(c))
    images = os.listdir(path)

    for i in images:
        try:
            image = Image.open(path + '\\' + i)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(c)
        except Exception as e:
            print(e)

data = np.array(data)
labels = np.array(labels)

#os.mkdir('training')
np.save('./training/data', data)
np.save('./training/target', labels)
'''
# --- allow_pickle=False error workaround when loading the data ---

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
data = np.load('./training/data.npy')
labels = np.load('./training/target.npy')
# restore np.load for future normal usage
np.load = np_load_old
# ---------
#print(data.shape)
#print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

'''
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # conf the model losses and metrics
epochs = 20
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test)) # train the model
model.save("./training/TSR.h5")





# accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
# --- Testing ---
model = load_model('./training/TSR.h5')


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data = []
    for img in imgs:
        img = os.path.join(root, img)
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    x_test = np.array(data)
    return x_test, label


testcsv = os.path.join(root, 'Test.csv')

x_test, label = testing(testcsv)
y_pred = model.predict_classes(x_test)
print("y_pred: ", y_pred)
print(accuracy_score(label, y_pred))


model = load_model('./training/TSR.h5')

# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }


def test_on_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))
    x_test = np.array(data)
    y_pred = model.predict_classes(x_test)
    return image, y_pred


plot, prediction = test_on_img(r'C:\Users\GHANEM\Desktop\OpenCV\data\GTSRB\Test\00900.png')
s = [str(i) for i in prediction]
a = int("".join(s))
print("Oredicted traffic sign is: ", classes[a])
plt.imshow(plot)
plt.show()