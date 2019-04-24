
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from image_preprocessing import data_transforms
from dl_classifier import set_checkpoint
from dl_classifier import model_arch
from dl_classifier import model_training
from dl_classifier import model_testing
import argparse

# The command line arguments 
def arguments_set():
  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str, help='The directory to the dataset ')
    parser.add_argument('--gpu', action='store_true', help = 'Use GPU to train the model')
    parser.add_argument('--arch', type=str, default='vgg16', choices = ['vgg16', 'vgg19'], help = 'The model architecture')
    parser.add_argument('--hidden_units', nargs='+', type=int, default = [512], help = 'The number of hidden units in the layers, You can enter more than one vlaue')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'The learning rate value')
    parser.add_argument('--epochs', type=int, default = 15, help = 'The number of epochs')
    parser.add_argument('--checkpoint_pth', type=str, default = './checkpoint.pth', help = 'The directory to save the checkpoint')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser_set = parser.parse_args()
    return parser_set

# Main function
def main():
    
    #arguments_set object
    arg_input = arguments_set()
    
    # Run on gpu
    if arg_input.gpu == True:
        device = 'cuda'
    else: 
         device = 'cpu'
    
    # Dataset pathes
    train_dir = arg_input.data_dir + '/train'
    valid_dir = arg_input.data_dir + '/valid'
    test_dir = arg_input.data_dir + '/test'

    # Dataset loading
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms['test'])
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    # The model output size is equal to the number of the classes
    output_size = len(train_datasets.classes)

    # Define the dataloaders
    dataloader = { 'train': torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle=True),
                  'validate': torch.utils.data.DataLoader(valid_datasets, batch_size = 64),
                  'test': torch.utils.data.DataLoader(test_datasets, batch_size = 64)}

    # Build the model
    model = model_arch(arg_input.arch, output_size, arg_input.hidden_units)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=arg_input.learning_rate)

    # Convert class to index and index to class
    model.class_to_idx = train_datasets.class_to_idx
    model.idx_to_class = dict((v,k) for k,v in model.class_to_idx.items())
    
    # Train the model
    model_training(model, dataloader, device, criterion, optimizer, arg_input.epochs)

    # Test the model
    model_testing(model, dataloader, device, criterion)
    
    # Save checkpoint
    set_checkpoint(model, arg_input.arch, output_size, optimizer, arg_input.epochs, arg_input.checkpoint_pth)

# Run the main function
if __name__ == "__main__":
    main()