import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np

from dataloader import Clima_dataset, custom_collate_fn
from conv_autoencoder import Conv_autoencoder
from utils import train_model_ae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    PCT_TRAINSET = 0.80
    LR = 0.1
    EPOCHS = 100

    input_path = '/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/'
    target_path = '/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED/'
    log_path = '/m100_work/ICT22_ESP_0/vblasone/rainfall-maps/.log/'

    with open(log_path+'log_ae.txt', 'w') as f:
        f.write(f'\nStarting on {device}.')

    # create the dataset
    dataset = Clima_dataset(input_path+'input.pkl', target_path+'target.pkl')

    # split into trainset and testset
    len_trainset = int(len(dataset) * PCT_TRAINSET)
    len_testset = len(dataset) - len_trainset

    with open(log_path+'log_ae.txt', 'a') as f:
        f.write(f'\nTrainset size = {len_trainset}, testset size = {len_testset}.')

    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset))

    # construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    # define the model
    conv_ae = Conv_autoencoder(input_size=25)

    # train the model
    
    loss_ae = nn.functional.mse_loss
    optimizer_ae =  torch.optim.Adam(conv_ae.parameters(), lr=LR, weight_decay=5e-4, momentum=.9)

    total_loss, loss_list = train_model_ae(conv_ae, dataloader=trainloader, loss_fn=loss_ae, optimizer=optimizer_ae, num_epochs=EPOCHS)

    np.savetxt('loss.csv', loss_list)