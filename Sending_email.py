import os
import smtplib
import imghdr
from email.message import EmailMessage
import cv2
import tensorflow as tf


CATEGORIES = ["Employee1", "Employee2", "Employee3", "unrecognized"]
test_file_name = 'C:/Users/User/Desktop/test.jpg'

sender_email = 'monitoringsiecineuronowe@gmail.com'
password = ''   #hidden

msg = EmailMessage()
msg['Subject'] = 'Alert'
msg['From'] = 'monitoringsiecineuronowe@gmail.com'
msg['To'] = 'monitoringsiecineuronowe@gmail.com'


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (112, 112))
    return new_array.reshape(-1, 112, 112, 1)

models = ["Model1.h5", "Model2.h5", "Model3.h5"]


def binary_prediction():
    counter = 0
    for i in range (3):
        model = tf.keras.models.load_model(models[i])
        prediction = model.predict(prepare(test_file_name))
        print(prediction)

        if (prediction == 0):
            counter = counter + 1
            i_value = i

    if counter == 0:
        print("The person was not recognized")
        msg.set_content('An unidentified person tried to enter the company')

        with open(test_file_name, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = 'An unidentified person'

    if counter == 1:
        if (i_value == 0):
            print("Recognized Employee1")

        if (i_value == 1):
            print("Recognized Employee2")

        if (i_value == 2):
            print("Recognized Employee3")

    if counter > 1:
        print("To verify")
        msg.set_content('Problem with verification')
        with open(test_file_name, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = 'Person to check'

    if counter != 1:
        msg.add_attachment(file_data, maintype='image',subtype=file_type,filename=file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)

def categorialical_prediction():
    model = tf.keras.models.load_model("Kategorie6.h5")

    prediction = model.predict(prepare(test_file_name))

    if prediction[0][0] > 0.5:
        print("Recognized Employee1")

    if prediction[0][1] > 0.5:
        print("Recognized Employee2")

    if prediction[0][2] > 0.5:
        print("Recognized Employee3")

    if prediction[0][3] > 0.5:
        print("The person was not recognized")
        msg.set_content('An unidentified person tried to enter the company')

        with open(test_file_name, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = 'An unidentified person'

            msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(sender_email, password)
                smtp.send_message(msg)


#categorialical_prediction()
binary_prediction()