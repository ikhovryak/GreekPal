import os

# ---------------------------------------
# PREPROCESS AND AUGMENT SETTINGS
# ---------------------------------------

COLLECTIVE_INPUT_MODE = True  # set to True if preprocessing multiple images, False if preprocessing one
DS_MACHINE_MODE = True # set to True if working on DS server, False if working locally 
ds_machine_path = os.getcwd() # os.path.join("home", 'ds', 'GreekPal', 'GreekPalLocal')
PATH_TO_MAIN_FOLDER = (ds_machine_path if DS_MACHINE_MODE else r'D:\Haverford\DigitalScholarship\GreekPalLocal')
NUM_TRAIN_FILES_DESIRED = 50 # number of files to augment in train and val each
NUM_VAL_FILES_DESIRED = NUM_TRAIN_FILES_DESIRED//2

OUTPUT_IMAGE_SIZE = 400

if COLLECTIVE_INPUT_MODE:
    PATH_TO_SOURCES = os.path.join(PATH_TO_MAIN_FOLDER, 'Sources', '*')
    PATH_TO_SAVE_TRAIN = os.path.join(PATH_TO_MAIN_FOLDER, 'Train')
    PATH_TO_SAVE_VAL = os.path.join(PATH_TO_MAIN_FOLDER, 'Val')

    print(os.path.isdir(PATH_TO_SAVE_TRAIN))
    print(os.path.isdir(PATH_TO_SAVE_VAL))
    if not os.path.isdir(PATH_TO_SAVE_TRAIN):
        os.mkdir(PATH_TO_SAVE_TRAIN)
    if not os.path.isdir(PATH_TO_SAVE_VAL):
        os.mkdir(PATH_TO_SAVE_VAL)
else:
    PATH_TO_INPUT_IMAGE = r''

# ---------------------------------------
# TRAIN AND RUN_TESTS SETTINGS
# ---------------------------------------
epochs = 10
PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), ('cur_model_' + str(epochs) +'epochs'))
