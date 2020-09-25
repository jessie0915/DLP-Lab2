# DLP-Lab2
## EEG Classification
#### Lab Objective
* In this lab, you will need to implement simple EEG classification models which are EEGNet, DeepConvNet[1] with BCI competition dataset. 
* Additionally, you need to try different kinds of activation function including『ReLU』,『Leaky ReLU』, 『ELU』.

#### Requirements
* Implement the EEGNet, DeepConvNet with three kinds of activation function including『ReLU』,『Leaky ReLU』, 『ELU』.
* In the experiment results, you have to show the highest accuracy (not loss) of two architectures with three kinds of activation functions.
* To visualize the accuracy trend, you need to plot each epoch accuracy (not loss) during training phase and testing phase.

#### Dataset
* BCI Competition III – IIIb
* [2 classes, 2 bipolar EEG channels]
* Reference: http://www.bbci.de/competition/iii/desc_IIIb.pdf

![Dataset](/picture/dataset.png "Dataset")

#### Prepare Data
* Training data: S4b_train.npz, X11b_train.npz
* Testing data: S4b_test.npz, X11b_test.npz
* To read the preprocessed data, refer to the “dataloader.py”.

![Prepare Data](/picture/prepare_data.png "Prepare Data")

#### Create Model - EEGNet

![EEGNet](/picture/EEGNet.png "EEGNet")
* Reference: Depthwise Separable Convolution
* https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

* EEGNet implementation details

![EEGNet implementation details](/picture/EEGNet_imp_details.png "EEGNet implementation details")

#### Create Model - DeepConvNet
* You need to implement the DeepConvNet architecture by using the following table, where C = 2, T = 750 and N = 2. The max norm term is ignorable.
* The input data has reshaped to [B, 1, C, T]

![DeepConvNet](/picture/DeepConvNet.png "DeepConvNet")

#### Activation Functions
* In the PyTorch framework, it is easy to implement the activation function.

![Activation Functions](/picture/activation_functions.png "Activation Functions")

#### Hyper Parameters
* Batch size= 64        
* Learning rate = 1e-2        
* Epochs = 150
* Optimizer: Adam      
* Loss function: torch.nn.CrossEntropyLoss()

* You can adjust the hyper-parameters according to your own ideas.

* If you use “nn.CrossEntropyLoss”, don’t add softmax after final fc layer because this criterion combines LogSoftMax and NLLLoss in one single class.

#### Result Comparison
* You have to show the highest accuracy (not loss) of two architectures with three kinds of activation functions.
* For example,

![Result Comparison](/picture/result_comparison.png "Result Comparison")
* To visualize the accuracy trend, you need to plot each epoch accuracy (not loss) during training phase and testing phase.
* In this part, you can use the matplotlib library to draw the graph. 
* For example,

![Result Comparison - Chart](/picture/result_comparison2.png "Result Comparison - Chart")



