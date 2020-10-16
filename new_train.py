import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from settings import PATH_TO_SAVE_TRAIN, PATH_TO_SAVE_VAL, epochs, OUTPUT_IMAGE_SIZE
import os

batch_size = 256

transformations = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


train_set = datasets.ImageFolder(PATH_TO_SAVE_TRAIN,  transform = transformations)
val_set = datasets.ImageFolder(PATH_TO_SAVE_VAL,  transform = transformations)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

input_size = 3*OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE
# hidden_sizes = [batch_size*2, batch_size]
hidden_sizes = [128, 64]



labels = os.listdir(PATH_TO_SAVE_TRAIN)
output_size = len(labels)

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
images = images.view(images.shape[0], -1)

if torch.cuda.is_available():
    logps = model(images.cuda()) #log probabilities
    loss = criterion(logps, labels.cuda()) #calculate the NLL loss
else:
    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
print('Initial weights - ', model[0].weight)

images, labels = next(iter(train_loader))
images.resize_(batch_size, 3*OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
if torch.cuda.is_available():
    output = model(images.cuda())
    loss = criterion(output, labels.cuda())
else:
    output = model(images)
    loss = criterion(output, labels)

loss.backward()
print('Gradient -', model[0].weight.grad)

# -------------------------------
# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)
# ------------------------------
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        if torch.cuda.is_available():
            output = model(images.cuda())
            loss = criterion(output, labels.cuda())
        else:
            output = model(images)
            loss = criterion(output, labels)
        #This is where the model learns by backpropagating
        loss.backward()
        #And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

saved_model_path = os.path.join(os.getcwd(), 'basic_model.pt')

torch.save(model, saved_model_path) 


correct_count, all_count = 0, 0
for images,labels in val_loader:
  for i in range(len(labels)):
#    print(len(images[i]))
    img = images[i].view(1, 480000) #OUTPUT_IMAGE_SIZE*OUTPUT_IMAGE_SIZE
    # Turn off gradients to speed up this part
    with torch.no_grad():
        
        logps = (model(img.cuda()) if torch.cuda.is_available() else model(img))

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = (list(ps.cpu().numpy()[0]) if torch.cuda.is_available() else list(ps.numpy()[0]))
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
