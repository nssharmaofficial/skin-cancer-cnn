# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:47:38 2022

@author: Nataša
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms

class Classifier:
    
    def __init__(self, backbone = "resnet", device = "cpu"):
        """Create an untrained classifier.

        Args:
            backbone: the string ("resnet","alexnet", "squeezenet", "simplecnn") that indicates which backbone network will be used.
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use.
        """

        self.num_outputs = 2                # number of classes
        self.net = None                     # NN - backbone + additional projection to output         
        self.device = torch.device(device)  # device where data will be moved
        self.data_mean = torch.zeros(3)     # mean of training data on the 3 channels (RGB)
        self.data_std = torch.ones(3)       # std of training data on the 3 channels (RGB)
        self.preprocess_train = None        # image preprocessing operations (when training)
        self.preprocess_eval = None         # image preprocessing operations (when evaluating)
    
           
       
        # case 1 - transfer learning (pretrained net + optional frozen weights)
        if backbone is not None and backbone in ['resnet', 'alexnet', 'squeezenet']:
            
            # pretrained = False → I will only take the architecture of the Resnet, the the next line
            # about weights should either disappear or I should change it into 'True' (as i dont have any learned weights)
            if backbone=='resnet': 
                self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            
            if backbone=='alexnet': 
                self.net = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True) 

            if backbone=='squeezenet': 
                self.net = torch.hub.load('pytorch/vision:v0.6.0', "squeezenet1_0", pretrained=True) 
            for param in self.net.parameters():
                param.requires_grad = True
    
            # adding a new (learnable) final classifier (fc)
            self.net.fc = nn.Linear(2048, self.num_outputs)
            
            # mean and std of the data on which the resnet was trained
            self.data_mean[:] = torch.tensor([0.485, 0.456, 0.406])
            self.data_std[:] = torch.tensor([0.229, 0.224, 0.225])
            
            # preprocessing operations to transform the input image accordingly to what resnet expects
            self.preprocess_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_mean, std=self.data_std),
            ])
            self.preprocess_eval = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_mean, std=self.data_std),
            ])
        
        
        # case 3 - SimpleCNN-based network
        elif backbone is not None and backbone == "simplecnn":
            self.net = nn.Sequential(
                
                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), # in_channels = 3 (RGB), out_channels=64 (random), padding = kernel_size/2
                # 32x(224x224)
                nn.ReLU(inplace=True),
                # 32x(224x224)
                nn.MaxPool2d(kernel_size=3, stride=2),
                # 32x(111x111)
                
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                # 64x(111x111)
                nn.ReLU(inplace=True),
                # 64x(111x111)
                nn.MaxPool2d(kernel_size=3, stride=2),
                # 64x(55x55)
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                # 128x(27x27)
                nn.ReLU(inplace=True),
                # 128x(27x27)
                nn.MaxPool2d(kernel_size=3, stride=2),
                # 128x(13x13)
                
                nn.Flatten(),
                # 2304 = 128*13*13
                
                # output size = 4096 → randomly chosen, can be any number
                nn.Linear(128 * 13 * 13, 2048),
                
                nn.ReLU(inplace=True),
                
                nn.Dropout(),
                
                nn.Linear(2048, self.num_outputs)
                )
            
 
            # preprocessing operations to transform the input image accordingly to what simplecnn expects
            # we have just chosen the resolution 32x32 but I can change it to something else as well
            self.preprocess_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_mean, std=self.data_std),
            ])
            self.preprocess_eval = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_mean, std=self.data_std)
            ])
            
            
        else:
            if backbone is not None:
                raise ValueError("Unknown backbone {}".format(str(backbone)))
            else:
                raise ValueError("Specify a bacckbone network")
                
        
        # save data_mean and data_std together with the net
        self.net.register_buffer("data_mean_buffer", self.data_mean)
        self.net.register_buffer("data_std_buffer", self.data_std)
        
        # move the network to the right device memory
        self.net.to(self.device)
        
        
    def save(self, file_name):
        """
        Save the classifier to file_name (net, data_mean, data_std)

        """
        torch.save(self.net.state_dict(), file_name)
    
    
    def load(self, file_name):
        """
        Load the classifier from file_name (net, data_mean, data_std)

        """
        self.net.load_state_dict(torch.load(file_name, map_location = self.device))
        self.net.to(self.device)
        
        # update mean & std to match the values in buffer
        for name, tensor in self.net.named_buffers():
            if name == "data_mean_buffer":
                self.data_mean[:] = tensor[:]
            elif name == "data_mean_std":
                self.data_std[:] = tensor[:]
    
    
    def forward(self, X):
        """
        Compute the outputs and logits of the network X
        """
        
        # outputs before applying the activation function
        logits = self.net(X)
        
        # outputs after applying the activation function
        outputs = torch.nn.functional.softmax(logits, dim=1)
        
        return outputs, logits
   
    
    @staticmethod
    def __loss(logits, labels):
        """Compute the loss function of the classifier.

        reduction='mean': the sum of the output will be divided by the number of elements in the output
    
        Args:
            logits: the (partial) outcome of the forward operation.
            labels: 1D tensor with the class labels.

        Returns:
            The value of the loss function.
        """

        total_loss = F.cross_entropy(logits, labels, reduction="mean")
        return total_loss
    
    
    @staticmethod
    def decision(outputs):
        """Given the tensor with the net outputs, compute the final decision of the classifier (class label).

        Args:
            outputs: the 2D tensor with the outputs of the net (each row is about an example).

        Returns:
            1D tensor with the main class IDs (for each example).
        """

        decided_classes = torch.argmax(outputs, dim=1)
        
        return decided_classes
    
    
    def __performance(self, outputs, labels):
        """Compute the accuracy in predicting the main classes.

        Args:
            outputs: the 2D tensor with the network outputs for a batch of samples (one example per row).
            labels: the 1D tensor with the expected labels.

        Returns:
            The accuracy in predicting the main classes.
        """

        # making a decision
        decided_classes = self.decision(outputs)

        # computing the accuracy of decisions
        right_predictions = torch.eq(decided_classes, labels)
        acc = torch.mean(right_predictions.to(torch.float) * 100.0).item()

        return acc
    
    
    def eval_classifier(self, data_set):
        """Evaluate the classifier (get the accuracy) on the given data set."""

        # if we are in 'train' mode, switch to 'eval'
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()  

        cpu_batch_outputs = []
        cpu_batch_labels = []

        # keeping off the autograd engine
        with torch.no_grad():  

            # loop on mini-batches to accumulate the network outputs
            for _, (X, Y) in enumerate(data_set):
                
                X = X.to(self.device)

                # network output on the current mini-batch (dont need logits)
                outputs, _ = self.forward(X)
                
                # accumulate outputs & labels
                cpu_batch_outputs.append(outputs.cpu())
                cpu_batch_labels.append(Y)

            # accuracy of the net on the whole dataset
            accuracy = self.__performance(torch.cat(cpu_batch_outputs, dim=0), torch.cat(cpu_batch_labels, dim=0))

        # restore the 'train' mode if needed
        if training_mode_originally_on:
            self.net.train()  

        return accuracy        
        

    def train_classifier(self, train_set, validation_set, batch_size, lr, epochs):
        """
        Trains the classifier and prints the accuracy of of both the training and validation set for each epoch
        and in the ends saves the best model as classifier.pth
        """
        
        self.net.train()
        
        # initialization
        best_val_acc = -1
        best_epoch = -1
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr)
        
        # loop on epochs
        for e in range(0, epochs):
            
            # accumulating mini-batch stats
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0
            
            for X,Y in train_set:
                
                # batch_size can differ from given (for example the last batch can be smaller)
                batch_num_train_examples = X.shape[0]
                epoch_num_train_examples += batch_num_train_examples
                
                X = X.to(self.device)
                Y = Y.to(self.device)
                
                # network output on current mini-batch
                outputs, logits = self.forward(X)
                
                # loss on current mini-batch
                loss = Classifier.__loss(logits, Y)
                
                # computing gradients (zero the previously computed)
                optimizer.zero_grad()
                loss.backward()
                
                # updating weights
                optimizer.step()
                
                """ Performance of the net on current train mini-batch"""
                with torch.no_grad():
                    
                    # switch to evaluation mode
                    self.net.eval()
                    
                    # accuracy
                    accuracy = self.__performance(outputs, Y)
                    
                    # accumulate accuracy and loss throughout whole set (running estimates) 
                    epoch_train_acc += accuracy * batch_num_train_examples
                    epoch_train_loss += loss.item() * batch_num_train_examples
                    
                    # going back to train mode
                    self.net.train()
                    
                    # printing BATCH related stats 
                    print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), accuracy))
                    
            
            # calculate the accuracy of current net in this epoch on validation set        
            val_acc = self.eval_classifier(validation_set)
            
            # save the model if validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                self.save("classifier.pth")
                
            # get estimation of the loss
            epoch_train_loss /= epoch_num_train_examples
            
            # printint EPOCH related stats
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                   + (", BEST!" if best_epoch == e + 1 else ""))
                  .format(e + 1, epochs, epoch_train_loss,
                          epoch_train_acc / epoch_num_train_examples, val_acc)) 
