import matplotlib.pyplot as plt
import os
from keras.preprocessing import image #modul zawierajacy narzedzia przetwarzajace obrazy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#paths to folders: base and the new one with processed images after augmentation

base_folder = 'C:/Users/User/Desktop/Employee1'
augmentation_folder = 'C:/Users/User/Desktop/Employee1_aug'

#file_names contains a list with data names in base folder with the whole path to the image
file_names = [os.path.join(base_folder, file_name) for file_name in os.listdir(base_folder)]

#function to generate transformations on the images
augmentation_data_generator = ImageDataGenerator(
    rescale = 1./255,
    rotation_range= 20,
    width_shift_range= 0.05,
    height_shift_range= 0.05,
    #shear_range= 0.2,
    #zoom_range= 0.1,
    #horizontal_flip= True,
    )

for image_path in file_names:
    img = image.load_img(image_path, grayscale = True, target_size = None) #loading an image in format: <PIL.Image.Image image mode=RGB size=92x112 at 0x14F97508400>

    tensor = image.img_to_array(img) #change the image into an Numpy array (112,92,1)

    tensor = tensor.reshape((1,) + tensor.shape) #change the shape to (1, 112, 92, 1)

    exit_of_the_loop = 0

    #flow instruction generates batches with randomly modified pictures, it has to be stopped, because the loop is working in infinity
    for batch in augmentation_data_generator.flow(
        tensor, #input data, numpy array of rank 4
        batch_size =32, #default:32
        save_to_dir = augmentation_folder, #saving in concrete directory
        save_prefix= 'Employee1', #prefix to use for saved pictures
        save_format= 'jpg'): #format of the photos

        exit_of_the_loop = exit_of_the_loop + 1
        if exit_of_the_loop % 10 == 0:
            break

    plt.show()