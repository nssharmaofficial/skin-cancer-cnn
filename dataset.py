# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:23:44 2022

@author: Nataša
"""
import torch
import os
from PIL import Image

class Dataset(torch.utils.data.Dataset):
        
    def __init__(self, path, empty_dataset=False):
        
        self.path = path        # path to the root of dataset file
        self.files = []         # names of dataset files
        self.images = []        # images loaded from dataset file
        self.labels = []        # labels of the main classes of each file
        
        self.preprocess = None
        
        if path is None:
            raise ValueError("You must specify the dataset path!")
        if not os.path.exists(path) or os.path.isfile(path):
            raise ValueError("Invalid data path: " + str(path))


        """ Counting number of classes (1 class = 1 folder)"""
        # list of the names of each folder in your dataset file ('data')
        folder_contents = os.listdir(self.path)
        main_class_dirs = [f for f in folder_contents    
                           if os.path.isdir(os.path.join(self.path, f)) and not f.endswith(".")]        
        
        # main_class_dirs = [benign malign]
        main_class_dirs = sorted(main_class_dirs) 
        
        # numbers of folders = numbers of classes
        self.main_class_count = len(main_class_dirs)
        
        
        """ Storing file names and labels """
        if not empty_dataset:
            
            # class index
            j= 0
            
            # iterate for each folder [benign malign]
            for main_class_name in main_class_dirs:
                
                # take the path for each such folder
                class_folder = os.path.join(self.path, main_class_name)
                
                # create a list of paths to each image in the folder
                folder_contents = os.listdir(class_folder)
                files = [os.path.join(class_folder, f) for f in folder_contents
                         if os.path.isfile(os.path.join(class_folder, f)) and f.endswith(".jpg")]
                
                # storing the file names in the dataset
                self.files.extend(files)
                
                # storing labels in the dataset → each file of the class inherit the labeling for such class
                self.labels.extend([j]*len(files))
                j += 1
                

    def __len__(self):
        """ The total number of examples in whole dataset"""
        return len(self.files)
    
    
    def set_preprocess_operation(self, custom_operation):
        """Define a custom operation to be applied to each image (the operation is assumed to return a tensor!)."""

        self.preprocess = custom_operation            
                
        
    def __getitem__(self,index):
        """ Load and return the next (image,label) pair from disk.
        
        instead of get_next → not asking to get NEXT picture but to picture with certain INDEX

        Args:
            index: the index of the element to be loaded.

        Returns:
            The pair (image,label).
        """
        
        # loading image using PIP
        image = Image.open(self.files[index]).convert('RGB')
        
        # preprocessing image (if needed)
        if not isinstance(image, torch.Tensor):  
            if self.preprocess is not None:
                image = self.preprocess(image)  
                
        # getting the label
        label = torch.tensor(self.labels[index], dtype=torch.long)
           
        return image, label
        
    
    def create_splits(self, proportions:list):
        """Create a given number of splits from the current dataset (stratified).

        Args:
            proportions: list with the fractions of data to store in each split (they must sum to 1).

        Returns:
            datasets = A list of Dataset objects (one Dataset per split).
        """
        
        # checking if proportions sum up to 1
        p = 0
        invalid_prop_found = False
        for prop in proportions:
            if prop <= 0.0:
                invalid_prop_found = True
                break
            p += prop
        if p!=1.0 or invalid_prop_found or len(proportions)==0:
            raise ValueError("Invalid proportions were provided - they must be positive and must sum to 1")
            
        """ Diving the data with respect to the main classes"""
        data_per_class = []
        
        # create list in list for each class
        for j in range(0, self.main_class_count):
            data_per_class.append([])
            
        # append all the images to list which belongs to their class
        for i in range(0, len(self.files)):
            data_per_class[self.labels[i]].append(i)
        
        """ Creating a list of (empty) Dataset object - one for each split"""
        num_splits = len(proportions)
        datasets = []
        for i in range(0, num_splits):
            datasets.append(Dataset(self.path, empty_dataset=True))
        
        """" Splitting the data """
        # for each class
        for j in range(0, self.main_class_count): 
            
            # index of the first element
            start = 0 
            
            # for each split
            for i in range(0, num_splits): # for each split
                p = proportions[i]
                
                # number of element to consider for class 'j'
                n = int(p*len(data_per_class[j])) 
                
                # for last class take the whole rest
                end = start+n if (i<num_splits-1) else len(data_per_class[j]) 
                
                # indices of samples to consider
                sample_ids = data_per_class[j][start:end]
                
                # adding them to the current split
                datasets[i].files.extend([self.files[s] for s in sample_ids])
                datasets[i].labels.extend([self.labels[s] for s in sample_ids])
                if len(self.images)>0:
                    datasets[i].images.extend([self.images[s] for s in sample_ids])
                    
                # moving the index
                start = end     
        
        return datasets