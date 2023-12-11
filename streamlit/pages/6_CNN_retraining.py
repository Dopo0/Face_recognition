import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import time



# Definition of the class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(12),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 112 #28 #56
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 56 #14 #28
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(48),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 56 #7 #14
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(48 * 14 * 14, out_features=1024),
            #nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, out_features=512),
            # nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=4)
        )

    def forward(self, x):
        x = self.convolutional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x


# Parameters Initialization
def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

# Save model
def save_model(model):
    with st.spinner('saving model...'):
        time.sleep(5)
        torch.save(model.state_dict(), 'obj/user_model2.pt')
    st.success('Saved!')
def main():
    st.title('CNN retraining')

    # Selecting devices available
    devices = ['cpu']
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    devices.append(device)
    if len(devices)>1:
        st.sidebar.selectbox("Select device:", devices)

    # Inputs from user
    epoch = st.sidebar.slider('Epoch',min_value=1, max_value=100, value=20, step=1 )
    save_models = st.sidebar.checkbox('Save model after training')

    # path to the folder with the images
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    relative_path_to_file = './data/faces_training'
    main_dir = os.path.join(project_root, relative_path_to_file)

    # getting data and transforming
    BATCH_SIZE = 5
    IMAGE_SIZE = (112, 112)
    train_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(degrees=15),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1,), (0.3,))])
    test_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1,), (0.3,))])

    train_path = './data/faces_training'
    test_path = './data/faces_training'
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    test_data = datasets.ImageFolder(test_path, transform=train_transform)

    train_iterator = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_iterator = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

    # Training and testing
    if st.sidebar.button("Train"):
        model = Network()
        model.to(device)
        model.apply(initialize_parameters)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(device)

        EPOCHS = epoch
        train_losses = []
        test_losses = []

        for EPOCH in range(EPOCHS):
            model.train()
            train_loss = 0

            for idx, (images, labels) in enumerate(train_iterator):

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                output = model(images)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            else:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in test_iterator:
                        images = images.to(device)
                        labels = labels.to(device)

                        log_probabilities = model(images)
                        test_loss += criterion(log_probabilities, labels)

                        probabilities = torch.exp(log_probabilities)
                        top_prob, top_class = probabilities.topk(1, dim=1)
                        predictions = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(predictions.type(torch.FloatTensor))

                train_losses.append(train_loss / len(train_iterator))
                test_losses.append(test_loss / len(test_iterator))

                st.write("Epoch: {}/{}  ".format(EPOCH + 1, EPOCHS),
                      "Training loss: {:.4f}  ".format(train_loss / len(train_iterator)),
                      "Testing loss: {:.4f}  ".format(test_loss / len(test_iterator)),
                      "Test accuracy: {:.2f}  ".format(accuracy / len(test_iterator)))
        else:
            final_acc = '{:.2f}'.format(accuracy / len(test_iterator))
            st.sidebar.success(f'Final accuracy: {final_acc}')

            if save_models:
                save_model(model)

if __name__ == '__main__':
    main()