from settings import *
import random
from scipy import ndarray
import cv2 as cv2
import numpy as np
import math
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from skimage.transform import SimilarityTransform
import os
import glob

def get_mask(image):
  result = image.copy()
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  lower = np.array([50, 100, 100])
  upper = np.array([140, 255, 255])
  mask = cv2.inRange(image, lower, upper)

  #find bounds of a symbol and crop it
  points = cv2.findNonZero(mask)
  x,y,w,h = cv2.boundingRect(points)
  crop_img = mask[y:y+h, x:x+w]
  
  result = cv2.bitwise_and(result, result, mask=mask)

  return crop_img

def get_resized_image(test_image):
  input_width = test_image.shape[1]
  input_height = test_image.shape[0]

  scale_factor = (OUTPUT_IMAGE_SIZE / max(input_height, input_width))*0.6

  needed_width = int(input_width * scale_factor)
  needed_height = int(input_height * scale_factor) 
  dim = (needed_width, needed_height)

  height = OUTPUT_IMAGE_SIZE
  width = OUTPUT_IMAGE_SIZE
  # resize image
  test_image = cv2.resize(test_image, dim, interpolation = cv2.INTER_AREA)

  blank_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)
  blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

  x_offset = int((width - test_image.shape[1])/2)
  y_offset = int((height - test_image.shape[0])/2)

  blank_image[ y_offset:y_offset+test_image.shape[0], x_offset:x_offset+test_image.shape[1]] = test_image

  

  return blank_image

def random_transformation1(img):
    tform = transform.SimilarityTransform(scale=1, rotation=math.pi/4, translation=(img.shape[0]/2, -100))
    rotated = transform.warp(img, tform)
    return rotated

def random_transformation(img):
    rows,cols = img.shape
    og_pt1 = [rows/6, cols/6]
    og_pt2 = [rows/6, cols - cols/6]
    og_pt3 = [rows - rows/6, cols/6]
    og_pts = [og_pt1, og_pt2, og_pt3]
    tf_pts = []
    for point in og_pts:
        shift1 = random.uniform(-1*rows/9, rows/9)
        shift2 = random.uniform(-1*cols/9, cols/9)
        tf_pt = [point[0]+shift1, point[1]+shift2]
        tf_pts.append(tf_pt)
    pts1 = np.float32(og_pts)
    pts2 = np.float32(tf_pts)
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows), borderValue=(0,0,0))
    return dst

def random_dilate(img):
    kernel = np.ones((5,5), np.uint8) 
    eroded = cv2.dilate(img, kernel, iterations=1)
    return eroded

def random_erode(img):
    kernel = np.ones((5,5), np.uint8) 
    eroded = cv2.erode(img, kernel, iterations=1)
    return eroded

def random_rotation(img):
    rows,cols = img.shape
    img_center = (cols / 2, rows / 2)
    random_degree = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D(img_center, random_degree, 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows), borderValue=(0,0,0))
    return rotated_image
    # pick a random degree of rotation between 25% on the left and 25% on the right
    #random_degree = random.uniform(-25, 25)
    #return sk.transform.rotate(image_array, random_degree)

def horizontal_warping(img):
    # Horizontal wave
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = 0
            a = float(random.randint(15, 18))
            b = random.randint(140, 160)
            offset_y = int(a * math.sin(2 * 3.14 * j / b))
            if i+offset_y < rows:
                img_output[i,j] = img[(i+offset_y)%rows,j]
            else:
                img_output[i,j] = 0

    return img_output

def vertical_warping(img):
    # Vertical wave
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            a = float(random.randint(22, 27))
            b = random.randint(170, 190)
            offset_x = int(a * math.sin(2 * 3.14 * i / b))
            offset_y = 0
            if j+offset_x < rows:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0

    return img_output

# dictionary of the transformations functions we defined earlier
available_transformations = {
    'transform': random_transformation,
    'transform2': random_transformation1,
    'erosion':random_erode,
    'rotate': random_rotation,
    "horizontal_warp": horizontal_warping,
    "vertical_warp": vertical_warping,
    #'noise': random_noise,
    #'horizontal_flip': horizontal_flip
}


def preprocess(image_path):
  image = cv2.imread(image_path)
  mask = get_mask(image)
  preprocessed = get_resized_image(mask)
  return preprocessed


def augment_images(path_to_file, num_files_needed):
    image_to_transform = cv2.imread(path_to_file)
    image_to_transform = cv2.cvtColor(image_to_transform, cv2.COLOR_BGR2GRAY)
    num_generated_files = 0
    # print(path_to_file)
    path_to_folder = os.path.dirname(path_to_file)

    num_existing_images = len(glob.glob(os.path.join(path_to_folder, "*")))
    # print("num_existing_images = ", num_existing_images)
    while num_generated_files <= num_files_needed:
        num_transformations = 0
        transformed_image = None
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
        
        new_file_path = '%s/augm_%s.png' % (path_to_folder, num_generated_files+num_existing_images)
        # write image to the disk
        sk.io.imsave(new_file_path, transformed_image, check_contrast=False)
        num_generated_files += 1

def get_label_name(image_path):
  # some image names have dashes at the beginning of the name, so use loop to get the first nonempty word 
  file_name = os.path.basename(image_path).split("-")
  for element in file_name:
    if len(element)>0:
      return element

def preprocess_and_save(image_path, label_name, image_full_name):
    preprocessed_image = preprocess(image_path)

    # create a folder for symbol in train, save to train
    os.chdir(PATH_TO_SAVE_TRAIN)
    path_in_dir = os.path.join(PATH_TO_SAVE_TRAIN, label_name)
    if not os.path.isdir(path_in_dir):
            os.mkdir(path_in_dir)

    new_file_name = image_full_name
    path_for_image = os.path.join(path_in_dir, new_file_name)
    cv2.imwrite(path_for_image, preprocessed_image)

    # create a folder for symbol in train, save to train
    os.chdir(PATH_TO_SAVE_VAL)
    path_in_dir = os.path.join(PATH_TO_SAVE_VAL, label_name)
    if not os.path.isdir(path_in_dir):
            os.mkdir(path_in_dir)

    new_file_name = image_full_name 
    path_for_image = os.path.join(path_in_dir, new_file_name)
    cv2.imwrite(path_for_image, preprocessed_image)


"""
def preprocess_augment_and_save(image_path, label_name):
  label_name = get_label_name(image_path)
  preprocessed_image = preprocess(image_path)

  # create a folder for each symbol in train, save and augment to train
  os.chdir(PATH_TO_SAVE_TRAIN)
  path_in_dir = os.path.join(PATH_TO_SAVE_TRAIN, label_name)
  print("path for directory = ", path_in_dir)
  if not os.path.isdir(path_in_dir):
        os.mkdir(path_in_dir)

  new_file_name = label_name + "_out.png"
  path_for_image = os.path.join(path_in_dir, new_file_name)
  cv2.imwrite(path_for_image, preprocessed_image)
  augment_images(path_in_dir, preprocessed_image, NUM_TRAIN_FILES_DESIRED)

  # create a folder for each symbol in val, save and augment to val
  os.chdir(PATH_TO_SAVE_TRAIN)
  path_in_dir = os.path.join(PATH_TO_SAVE_VAL, label_name)
  if not os.path.isdir(path_in_dir):
        os.mkdir(path_in_dir)

  new_file_name = label_name + "_out.png"
  path_for_image = os.path.join(path_in_dir, new_file_name)
  cv2.imwrite(path_for_image, preprocessed_image)
  augment_images(path_in_dir, preprocessed_image, NUM_VAL_FILES_DESIRED)

"""
