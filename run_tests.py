import torch
import os
import numpy as np
import cv2 as cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from settings import PATH_TO_SAVE_TRAIN, PATH_TO_SAVED_MODEL, DS_MACHINE_MODE, OUTPUT_IMAGE_SIZE


model = torch.load(PATH_TO_SAVED_MODEL)

model.eval()

transformations = transforms.Compose([
    # transforms.Resize(100),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# -----------------------------------
# Step 5: Actually Using our model
# -----------------------------------

    # Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path).convert('RGB')
    # img = cv2.imread(image_path, 0)

    # Get the dimensions of the image
    width, height = img.size
    # width = img.shape[1]
    # height = img.shape[0]

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 100px

    dim = (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)
    # test_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    
    # img = img.resize((100, int(100*(height/width))) if width < height else (int(100*(width/height)), 100))
    img = img.resize((OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
    # Get the dimensions of the new image size
    # width = img.shape[1]
    # height = img.shape[0]
    width, height = img.size

    """
    # Set the coordinates to do a center crop of 224 x 224
    # left = (width - 224)/2
    # top = (height - 224)/2
    # right = (width + 224)/2
    # bottom = (height + 224)/2
    # img = img.crop((left, top, right, bottom))
    """

    # Turn image into numpy array
    img = np.array(img)

    # img = np.reshape(img,img.shape+(1,))

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img/100

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]

    # Turn into a torch tensor
    if DS_MACHINE_MODE:
        image = torch.from_numpy(img).cuda()
    else:
        image = torch.from_numpy(img)

    image = image.float()
    return image


# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(5, dim=1)
    # probs, classes = output.topk(5, dim=1)

    # return probs.item(), classes.item()
    print(classes)
    return classes.tolist()[0]


def show_image(image):
    # Convert image to numpy
    image = image.numpy()

    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    # Print the image
    # fig = plt.figure(figsize=(25, 4))
    return np.transpose(image[0], (1, 2, 0))


def run_tests(test_images, axs):
    i = 0
    for test_image in test_images:
        path_to_test_image = os.path.join(os.getcwd(), 'Tests')
        original_image_path = os.path.join(path_to_test_image, (test_image + ".png"))
        print(original_image_path)
        image = process_image(original_image_path)
        # Give image to model to predict output
        top_prob, top_class = predict(image, model)
        top_class_label = train_set.classes[top_class]
        # Show the image
        # show_image(image)
        # Print the results
        print("Given image of class ", test_image)
        print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class_label, "\n")
        original_image = Image.open(original_image_path).convert('RGB')
        predicted_image = Image.open(os.path.join(PATH_TO_SAVE_TRAIN, top_class_label, (top_class_label+"_out.png"))).convert('RGB')
        axs[i, 0].imshow(original_image)
        axs[i, 1].imshow(predicted_image)
        axs[i, 0].set_title(top_prob*100)
        i+=1

def run_one_test(test_image):
    path_to_test_image = os.path.join(os.getcwd(), 'Tests')
    original_image_path = os.path.join(path_to_test_image, (test_image + ".png"))
    image = process_image(original_image_path)
    top_class = predict(image, model)
    for top in top_class:
        top_class_label = train_set.classes[top]
        print(top_class_label) 

# Now RUN:
# Process Image
train_set = datasets.ImageFolder(PATH_TO_SAVE_TRAIN)

print("train set classes = ", train_set.classes)

test_images = ["afrodita", "afrodita2", "kai", "yap2", "augm_14"]
fig, axs = plt.subplots(len(test_images), 2)
for test_image in test_images:
	print(test_image)
	run_one_test(test_image)
	print()
# run_one_test("augm_14")
# run_tests(test_images, axs)
plt.savefig(os.path.basename(PATH_TO_SAVED_MODEL)+".png")
plt.show()
# Show Image
