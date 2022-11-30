# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:48:11 2022

@author: Nataša
"""

# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from PIL import Image
import os
import argparse

from dataset import Dataset
from classifier import Classifier

    
def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=['train', 'eval'],
                        help='train or evaluate the classifier')
    parser.add_argument('dataset_folder', type=str,
                        help='training data set folder or evaluation data set folder')
    parser.add_argument('--splits', type=str, default='0.7-0.15-0.15',
                        help='fraction of data to be used in train, val, test set (default: 0.7-0.15-0.15)')
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'alexnet', 'squeezenet', 'simplecnn'],
                        help='backbone network for feature extraction (default: resnet)"')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (Adam) (default: 0.001)')
    parser.add_argument('--workers', type=int, default=3,
                        help='number of working units used to load the data (default: 3)')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parsed_arguments = parser.parse_args()

    # converting split fraction string to a list of floating point values ('0.7-0.15-0.15' => [0.7, 0.15, 0.15])
    splits_string = str(parsed_arguments.splits)
    fractions_string = splits_string.split('-')
    if len(fractions_string) != 3:
        raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
    else:
        splits = []
        frac_sum = 0.
        for fraction in fractions_string:
            try:
                splits.append(float(fraction))
                frac_sum += splits[-1]
            except ValueError:
                raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
        if frac_sum != 1.0:
            raise ValueError("Invalid split fractions were provided. They must sum to 1.")

    # updating the 'splits' argument
    parsed_arguments.splits = splits

    return parsed_arguments
    

if __name__ == "__main__":
    args = parse_command_line_arguments()

    # print the arguments and hyperparameters
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    if args.mode == 'train':
        print("Training the classifier...")

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # preparing dataset
        _data_set = Dataset(args.dataset_folder)

        # splitting dataset
        [_train_set, _val_set, _test_set] = _data_set.create_splits(args.splits)

        # adding classifier-specific preprocessing operations
        _train_set.set_preprocess_operation(_classifier.preprocess_train)
        _val_set.set_preprocess_operation(_classifier.preprocess_eval)
        _test_set.set_preprocess_operation(_classifier.preprocess_eval)


        # converting datasets into data loaders
        # takes dataset and provides facitilites to get minibatches of data
        _train_set = torch.utils.data.DataLoader(_train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers)

        _val_set = torch.utils.data.DataLoader(_val_set, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers)

        _test_set = torch.utils.data.DataLoader(_test_set, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers)

        # training the classifier
        _classifier.train_classifier(_train_set, _val_set, args.batch_size, args.lr, args.epochs)

        # loading the model that yielded the best results in the validation data (during the training epochs)
        print("Training complete, loading the best found model...")
        _classifier.load('classifier.pth')

        # computing the performance of the final model in the prepared data splits
        print("Evaluating the classifier...")
        _train_acc = _classifier.eval_classifier(_train_set)
        _val_acc = _classifier.eval_classifier(_val_set)
        _test_acc = _classifier.eval_classifier(_test_set)

        print("train set:\tacc={0:.2f}".format(_train_acc))
        print("val set:\tacc={0:.2f}".format(_val_acc))
        print("test set:\tacc={0:.2f}".format(_test_acc))


    elif args.mode == 'eval':
        print("Evaluating the classifier...")

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # loading the classifier
        _classifier.load('classifier.pth')

        # preparing the data (but not splitting)
        _data_set = Dataset(args.dataset_folder)

        # adding classifier-specific preprocessing operations
        _data_set.set_preprocess_operation(_classifier.preprocess_eval)

        # converting dataset into a dataloader
        _data_set = torch.utils.data.DataLoader(_data_set, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers)

        # not training the classifier  !!!
        
        # only computing the classifier performance
        _acc = _classifier.eval_classifier(_data_set)

        print("acc={0:.2f}".format(_acc))

                    
                
                


        
        
        
            
            
                
    
    
    
    
    
        