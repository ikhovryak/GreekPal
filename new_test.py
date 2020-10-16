import torch
import os
import numpy as np
import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from settings import *

path =  os.path.join(os.getcwd(), 'basic_model.pt')
model = torch.load(path)

transformations = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
batch_size = 256

train_set = datasets.ImageFolder(PATH_TO_SAVE_TRAIN, transform = transformations)
val_set = datasets.ImageFolder(PATH_TO_SAVE_VAL, transform = transformations)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


images, labels = next(iter(val_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = images[0].view(1, 3*OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE)


loader = transforms.Compose([
                            #transforms.Resize(100),
                            transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    # image = Variable(image, requires_grad=True)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return (image.cuda() if torch.cuda.is_available() else image)  #assumes that you're using GPU

# image = image_loader(PATH TO IMAGE)

# your_trained_net(image)

image_name="kai"
image_path = os.path.join(os.getcwd(), 'Tests', (image_name + ".png"))


def my_image_loader(image_path):
    img = Image.open(image_path)
    img = img.resize((OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
    img = np.array(img)
    print(img.shape)
    img = img.transpose((2, 0, 1))
    img = img/OUTPUT_IMAGE_SIZE
    print(img.shape)

    if torch.cuda.is_available():
        img = torch.from_numpy(img).cuda()
        
    else:
        img = torch.from_numpy(img)

img = Image.open(image_path)
img = img.resize((OUTPUT_IMAGE_SIZE,OUTPUT_IMAGE_SIZE))
img = np.array(img)
img = img.transpose((2, 0, 1))
img = cv2.bitwise_not(img)

img = loader(img)
# print(img.shape)
img = img.resize(1, 3*OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE)



# Turn off gradients to speed up this part
with torch.no_grad():
    logps = (model(img.cuda()) if torch.cuda.is_available() else model(img))

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = (list(ps.cpu().numpy()[0]) if torch.cuda.is_available() else list(ps.numpy()[0]))


train_labels = train_set.classes

# print("Correct symbol = ", train_labels[labels[0]])
print("Correct symbol = ", image_name)


print("Predicted symbol = ", train_labels[probab.index(max(probab))], probab.index(max(probab)))

# print("ps =", probab)

probabilities = {}
ordered_probabilities = sorted(probab, reverse=True)
# ordered_probabilities.sort(reverse=True)

for i in range(10):
    prob_percent = ordered_probabilities[i]
    prob_index = probab.index(prob_percent)
    prob_label = train_labels[prob_index]
    probabilities[prob_label] = prob_percent

print(probabilities)

# print("All labels: ", train_labels)

# view_classify(img.view(1, 3*OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE), ps)


