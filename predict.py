

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from dl_classifier import get_checkpoint
from image_preprocessing import process_image
import argparse
import json


# The command line arguments 
def arguments_set():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', help='Use GPU to train the model')
    parser.add_argument('--top_k', type=int, default = 1, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, help='The path of JSON that converts class number to the category name')
    parser.add_argument('img_path', type=str, help='The input image path')
    parser.add_argument('checkpoint', type=str, help='The path to a saved the checkpoint')
    parser_set = parser.parse_args()
    return parser_set


# Main  function 
def main():
    
    #arguments_set object
    arg_input = arguments_set()
    
   # Get checkpoint
    model = get_checkpoint(arg_input.checkpoint)
    
    # Run on gpu
    if arg_input.gpu == True:
        device = 'cuda'
    else: 
         device = 'cpu'
    
    model = model.to(device)
    
    # Preprocess images..
    processed_img = process_image(arg_input.img_path)
    processed_img = torch.unsqueeze(torch.from_numpy(processed_img).type(torch.FloatTensor),0)
    
    processed_img = processed_img.to(device)
    
    # Set evaluation mode for predicting
    model.eval()
    
    with torch.no_grad():
        log_ps = model.forward(processed_img)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(arg_input.top_k)
    
    top_p = top_p.data.cpu().numpy()[0]
    top_class = top_class.cpu().data.numpy()[0]
    
    classes = []
    for i in top_class:
        classes.append(model.idx_to_class[i])
    
    # If category_names value is set
    categories = {}
    categories_names = []
    if arg_input.category_names:
        with open(arg_input.category_names, 'r') as file:
            categories = json.load(file)
        for c in classes:
            categories_names.append(categories[str(c)])    
    
    # Print results
    if arg_input.top_k == 1:
        print('The top 1 most likely class is:')
    else:
        print('The top {} most likely classes are:'.format(arg_input.top_k))
        
    print('classes >> probability of belonging to the class')    
    
    for k in range(arg_input.top_k):
        if categories:
            print('{} >> {:.3f}'.format(categories_names[k], top_p[k]))
        else:
            print('{} >> {:.3f}'.format(classes[k], top_p[k]))

# Run the main function
if __name__ == "__main__":
    main()