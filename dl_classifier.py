# This python file contains all functions for creating, training and testing the model. It also contains set and get checkpoints functions  

import torch
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F

# Class deep learning network
class DLNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        
        super().__init__()
        
        first_layer = [nn.Linear(input_size, hidden_layers[0])]
        self.hidden_layers = nn.ModuleList(first_layer)
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(l1, l2) for l1, l2 in layer_sizes])
    
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p = 0.4)
        
    def forward(self, x):
        
        # Using relu activation function and dropout regularization in the hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        x = self.output(x)
        log = F.log_softmax(x, dim=1)
        
        return log

# Creating the deep learning model
def model_arch(arch, output_size, hidden_layers):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # fix the parameters of the pretrained model
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = DLNetwork(model.classifier[0].in_features, output_size, hidden_layers)
   
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        # fix the parameters of the pretrained model
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = DLNetwork(model.classifier[0].in_features, output_size, hidden_layers) 
  
    else:
        print ("The application does not use {} model. Please select vgg16 or vgg19.".format(arch)) 
        
    return model


def model_training(model, dataloader, device, criterion, optimizer, epochs):

    print("Model Training..")
    model = model.to(device)
    
    # Set model to train mode
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0
        # Model traiining
        for inputs, labels in dataloader['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                #Set to evaluation mode
                model.eval()
                
                for inputs, labels in dataloader['validate']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)
                    valid_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            
        #Set to train mode
        model.train()   
        
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss = {:.3f}".format(running_loss/len(dataloader['train'])),
              "Validation Loss = {:.3f}".format(valid_loss/len(dataloader['validate'])),
              "Validation Accuracy = {:.3f}".format(accuracy/len(dataloader['validate'])))
            
    print('The training is done!')
    
def model_testing(model, dataloader, device, criterion):
    
    print('Model Testing.....')
    
    test_loss = 0
    accuracy = 0
            
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        #Set to evaluation mode
        model.eval()
        
        for images, labels in dataloader['test']:
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels)
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        
    #Set to train mode
    model.train()   
    accuracy = accuracy/len(dataloader['test'])
    loss = test_loss/len(dataloader['test'])
    
    print("Testing Accuracy = {:.3f}".format(accuracy))
    print("Testing Loss = {:.3f}".format(loss))

def set_checkpoint(model, arch, output_size, optimizer, epoch, file_path):
    
    hidden_layers = []
    for layer in model.classifier.hidden_layers:
        hidden_layers.append(layer.out_features) 
        
    checkpoint_data = {
                  'class_to_idx': model.class_to_idx,
                  'idx_to_class': model.idx_to_class,
                  'state_dict': model.state_dict(),
                  'arch': arch,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'optimizer': optimizer.state_dict,
                  'epoch': epoch
                 }
    torch.save(checkpoint_data, file_path)


def get_checkpoint(filepath):
   
    checkpoint_data = torch.load(filepath)

    model = model_arch(checkpoint_data['arch'], checkpoint_data['output_size'], checkpoint_data['hidden_layers'])
    
    model.class_to_idx = checkpoint_data['class_to_idx']
    model.idx_to_class = checkpoint_data['idx_to_class']
    model.load_state_dict(checkpoint_data['state_dict'])
    
    return model