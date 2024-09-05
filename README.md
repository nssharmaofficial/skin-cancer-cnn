# Skin cancer CNN

## Motivation

According to the World Cancer Research Fund, among all cancers, skin cancer is the 19th most common. Among all skin cancers, malignant and benign are the deadliest. A malignant tumor is a type of cancerous tumor that spreads and expands in a patient's body. They can infiltrate other tissues and organs and develop and spread unchecked. The importance of detecting and treating cancer in early malignant skin growth cannot be overstated. On the contrary, a benign tumor has the capability to develop, but it is not going to spread. When it comes to benign skin growths, knowing the common signs and symptoms of those that could be malignant is critical, as is seeking medical attention when skin growths show suspect.

## Dataset Description

You can download the dataset [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

Dataset overview:

| **Class**   | **Examples** |
|-------------|--------------|
| Malignant   | 6602         |
| Benign      | 7300         |

## Accuracy of different nets with default parameters

| **Set**       | **Resnet50** | **Alexnet** | **Squeezenet** | **Custom CNN** |
|---------------|--------------|-------------|----------------|----------------|
| Train         | 86.94        | 86.50       | 85.13          | 84.25          |
| Validation    | 86.62        | 86.52       | 86.14          | 86.24          |
| Test          | 87.25        | 84.52       | 87.54          | 84.47          |

## CNN architecture

### Transfer learning

Pretrained nets used in transfer learning:
- [Resnet50](https://pytorch.org/hub/pytorch_vision_resnet/)
- [Alexnet](https://pytorch.org/hub/pytorch_vision_alexnet/)
- [Squeezenet](https://pytorch.org/hub/pytorch_vision_squeezenet/)

With final linear classifier:

```python
self.net.fc = nn.Linear(2048, self.num_outputs)
```

### Custom CNN 

3 convolutional layers followed by linear classifier:

```python
self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Flatten(),
                nn.Linear(128 * 13 * 13, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(2048, self.num_outputs)
           )
```

## How to use this code

Clone this repository:

```bash
git clone https://github.com/natasabrisudova/SkinCancer-CNN
```

Then run:

```bash
# example to evaluate the net on the data (dir 'data' with folders: malignant, benign)
$ skin_cancer.py eval data 
```





