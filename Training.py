from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import random
from tensorflow.keras import layers
import numpy as np
import os
from keras import models
from keras.preprocessing import image #module with tools to process images


def data_loading():
    #Global variables
    global data,labels

    #Enter the path of your image data folder
    image_data_folder_path = 'C:/Users/User/Desktop/Employee1'

    data = []
    labels = []

    #grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(image_data_folder_path)))

    #total number images
    total_number_of_images = len(imagePaths)
    print("\n")
    print("Total number of images----->",total_number_of_images)

    #randomly shuffle all the images filenames
    random.shuffle(imagePaths)

    print("Data processing...")
    #loop over the shuffled input images
    for imagePath in imagePaths:

        img = image.load_img(imagePath, color_mode = "grayscale", target_size=None)
        tensor = image.img_to_array(img)

        #Append each image data 1D array to the data list
        data.append(tensor)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    #scale the raw pixel intensities to the range [0, 1]
    #convert the data and label list to numpy array
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("Data processing finished")

def data_split():

    global trainX,testX,trainY,testY

    print("Data splitting...")

    #partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    #train_test_split is a scikit-learn's function which helps us to split train and test images kept in the same folders
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

    print("Number of training images--->", len(trainX), ",", "Number of training labels--->", len(trainY))
    print("Number of testing images--->", len(testX), ",", "Number of testing labels--->", len(testY))

    lb = preprocessing.LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    print("\n")
    print("Classes found to train", )
    train_classes = lb.classes_
    print(train_classes)
    binary_rep_each_class = lb.transform(train_classes)
    print("Binary representation of each class")
    print(binary_rep_each_class)
    print("\n")

def plot():
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Comparison of the recognition accuracy of the training and validation set")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy[%]")
    plt.legend()
    plt.show()

data_loading()
data_split()


EPOCHS = 10

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation = 'relu', input_shape = (112, 112, 1)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(4, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=50)

plot()



acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']


epochs = range(1, len(acc)+1)
plt.figure(2)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.save('Model1.h5')