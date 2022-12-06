# SkinCancer-CNN
 Skin cancer classification in pytorch using CNN

![](https://api.visitorbadge.io/api/VisitorHit?user=natasabrisudova&repo=SkinCancer-CNN&countColor=%237B1E7A)

### Things to add:
- more pre-trained models
- printing execution time
- printing graphs for accuracy and loss

<br>

## Motivation

According to the World Cancer Research Fund, among all cancers, skin cancer is the 19th most common. Among all skin cancers, malignant and benign are the deadliest. A malignant tumor is a type of cancerous tumor that spreads and expands in a patient's body. They can infiltrate other tissues and organs and develop and spread unchecked. The importance of detecting and treating cancer in early malignant skin growth cannot be overstated. On the contrary, a benign tumor has the capability to develop, but it is not going to spread. When it comes to benign skin growths, knowing the common signs and symptoms of those that could be malignant is critical, as is seeking medical attention when skin growths show suspect.

<br>

## Dataset Description

You can download the dataset [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

<p> Dataset overview:</p>

<table>
  <tr>
    <td colspan="3"></td>
  </tr>
  <tr>
    <td><b>Class</b></td>
    <td><b>Examples</b></td>
  </tr>
  <tr>
    <td>Malignant</td>
    <td>6602</td>
  </tr>
  <tr>
    <td>Benign</td>
    <td>7300</td>
  </tr>
</table>

<br>

## Accuracy of different nets with default parameters

<table>
  <tr>
    <td colspan="5"></td>
  </tr>
  <tr>
    <td><b>Set</b></td>
    <td><b>Resnet50</b></td>
    <td><b>Alexnet</b></td>
    <td><b>Squeezenet</b></td>
    <td><b>Custom CNN</b></td>
  </tr>
  <tr>
    <td>Train</td>
    <td>86.94</td>
    <td>86.50</td>
    <td>85.13</td>
    <td>84.25</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>86.62</td>
    <td>86.52</td>
    <td>86.14</td>
    <td>86.24</td>
  </tr>
    <tr>
    <td>Test</td>
    <td>87.25</td>
    <td>84.52</td>
    <td>87.54</td>
    <td>84.47</td>
  </tr>
</table>

<br>

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

<br>

## How to use this code

You'll need [Git](https://git-scm.com) to be installed on your computer.
```
# Clone this repository
$ git clone https://github.com/natasabrisudova/SkinCancer-CNN
```

<br>

Then in command prompt run:
```
# example to evaluate the net on the data (dir 'data' with folders: malignant, benign)
$ skin_cancer.py eval data 
```





