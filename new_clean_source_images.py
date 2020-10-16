import os
from settings import *
import imghdr
import glob
from new_preprocess_and_augment import augment_images, preprocess_and_save
import cv2 as cv2
import numpy as np
"""
PATH_TO_SOURCES is the directory with all the manuscripts
"""

# os.chdir(PATH_TO_SOURCES)

def get_label_name(image_path):
  # some image names have dashes at the beginning of the name, so use loop to get the first nonempty word 
  file_name = os.path.basename(image_path).split("-")
  for element in file_name:
    if len(element)>0:
      return element

def get_full_image_name(image_path):
    return os.path.basename(image_path)

image_types = {'jpg', 'png', 'jpeg', 'PNG', 'JPEG'}

newly_updated_labels = set()

# mans = r'D:\\Haverford\\DigitalScholarship\\GreekPalLocal\\Sources\\RomeVat.pal.gr.270'
# preprocess al the new images that haven't been preprocessed before
# print(glob.glob(os.path.join(mans, "*")))
for manuscript in glob.glob(PATH_TO_SOURCES):

    for image_path in glob.glob(os.path.join(manuscript, "*")):

        file_type = imghdr.what(image_path)
        # if the file is a training or highlighted image
        if file_type in image_types and ("-t-" in image_path or "-h-" in image_path): 
            image_name = get_full_image_name(image_path)
            image_label = get_label_name(image_path)
            path_to_symbol_in_train = os.path.join(PATH_TO_SAVE_TRAIN, image_label)
            # if the image hasn't been processed before
            if image_name not in glob.glob(path_to_symbol_in_train):
                print(image_path)
                preprocess_and_save(image_path, image_label, image_name)  # just preprocess and save, no augmentation
                newly_updated_labels.add(image_label)
        


# augment all the new images that haven't been preprocessed before
# for each label that have been updated:
print("newly_upd_labels = ", newly_updated_labels)
for label in newly_updated_labels:
#   - remove all the augmented files in train except for the originals
    path_to_symbol_in_train = os.path.join(PATH_TO_SAVE_TRAIN, label, "*")
    
    train_files = glob.glob(path_to_symbol_in_train)
    # print("all train files = ", train_files)
    num_original_train_images = 0
    for image in train_files:
        # print("checking image = ",image )
        if ("-t-" in image or "-h-" in image):
            num_original_train_images+=1
            
        else:
            os.remove(image)
        

    print()
#   - remove all the augmented files in val except for the originals
    path_to_symbol_in_val = os.path.join(PATH_TO_SAVE_VAL, label, "*")
    val_files = glob.glob(path_to_symbol_in_val)
    num_original_val_images = 0
    for image in val_files:
       

        if ("-t-" in image or "-h-" in image):
            num_original_val_images+=1
            
        else:
            os.remove(image)
    
#   - calculate the even number of files to augment each original symbol in label
    num_needed_augmented_train = NUM_TRAIN_FILES_DESIRED//num_original_train_images
    num_needed_augmented_val = NUM_VAL_FILES_DESIRED//num_original_val_images
    print("num_needed_augmented_train = ", num_needed_augmented_train)
    print("num_needed_augmented_val = ", num_needed_augmented_val)

#   - augment each original symbol in both train and val
    # print(train_files)
    train_files = glob.glob(path_to_symbol_in_train)
    for image in train_files:
        if file_type in image_types and ("-t-" in image or "-h-" in image):
            # augment function
            augment_images(image, num_needed_augmented_train)
    
    val_files = glob.glob(path_to_symbol_in_val)
    for image in val_files:
        if file_type in image_types and ("-t-" in image or "-h-" in image):
            # augment function
            augment_images(image, num_needed_augmented_val)



"""
Idea:
- add all original images first, get them into separate folders, save names as we go
- for each added name:
    - go into the folder, calculate the number of existing originals in each class
    - calculate the needed amount of augmented images
    - augment images 
"""

