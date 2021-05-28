# Train models on MNIST Dataset to achieve 99.4 validation accuracy

## Goal To achieve

The activity that was performed using MNIST data was to achieve a validation accuracy of atleast 99.4% within 20 epochs of training the model and ensuring the number of parameters are less than 20,000. The purpose of this activity was to understand how we can consider different types of network architectures, considering different loss functions can help us achieve the desired result.

Below we have explained in detail on how we achieved the goal and what was our findings as much detailed possible for us.

## Network Architectures Used

To test out what was the best approach, we used 6 different network architectures. The architecture diagram along with the model summary are listed below:

### Network - Vanilla

Vanilla Network: **Architecture Diagram**

![n1arch](./../images/network_1.png)

In the Vanilla architecture, we have a base structure which involves the convolution layers followed by batch normalization involving padding of 1. Once the receptive field reaches 11, we add a maxpool layer and add a transition block. This is followed by 3 more convolution and batch normalization blocks without any paddings on the convolution blocks. Once the number of pixels in each channel reaches 8x8, we add a GAP layer to reduce it to a 1D consolidated kernel. This is followed by a Linear layer leading towards a log softmax output.

The detailed number of parameters that are being used in the Network are listed below.

Vanilla Network: **Model parameters**

![n1param](./../images/network1_parameters.png)

### Network - With Dropout

Network With Dropout: **Architecture Diagram**

![n4arch](./../images/network_1.png)

In the network architecture, most of the structure is similar to the Vanilla network architecture. The difference between the two network is that, after the first layer, instead of having a batch normalization, we used a dropout. The reason why dropout was used here is that, the number of pixels is pretty high and the information is spread out. Loosing few pixels from the input data acts in some format like a data augmentation strategy.

The detailed number of parameters that are being used in the Network are listed below.

Network With Dropout: **Model parameters**

![n4param](./../images/network1_parameters.png)

### Network - With GAP and Transition Layer

Network With GAP and Transition Layer: **Architecture Diagram**

![n2arch](./../images/network_1.png)

In the network architecture, most of the structure is similar to the network with dropout architecture. We also added a transition block after the last convolution layer to reduce the usage on a fully connected layer.

The detailed number of parameters that are being used in the Network are listed below.

Network With GAP and Transition Layer: **Model parameters**

![n2param](./../images/network1_parameters.png)

### Network - With GAP and Transition Layer without FC

Network GAP and Transition Layer without FC: **Architecture Diagram**

![n3arch](./../images/network_1.png)

In the network architecture, most of the structure is similar to the network with GAP and transition layer architecture. We removed the fully connected layer to try how a fully convolutional layer would behave.

The detailed number of parameters that are being used in the Network are listed below.

Network GAP and Transition Layer without FC: **Model parameters**

![n3param](./../images/network1_parameters.png)

## Training the models

|Model Name|No. of Parameters|Learning Rate Scheduler|Data Augmentation|Batch Size|Training Accuracy|Test Accuracy|
|----------|-----------------|-----------------------|-----------------|----------|-----------------|-------------|
|Batch Norm + FC + GAP|9,186|None|None|64|||
|Batch Norm + FC + Dropout + GAP|9,178|None|None|64|||
|Batch Norm + FC + Transition + GAP|9,320|None|None|64|||
|Batch Norm + Transition + GAP|9,210|None|None|64|||
|Batch Norm + FC + GAP|9,186|StepLR|None|64|||
|Batch Norm + FC + Dropout + GAP|9,178|StepLR|None|64|||
|Batch Norm + FC + Transition + GAP|9,320|StepLR|None|64|||
|Batch Norm + Transition + GAP|9,210|StepLR|None|64|||
|Batch Norm + FC + GAP|9,186|ReduceLROnPlateau|None|64|||
|Batch Norm + FC + Dropout + GAP|9,178|ReduceLROnPlateau|None|64|||
|Batch Norm + FC + Transition + GAP|9,320|ReduceLROnPlateau|None|64|||
|Batch Norm + Transition + GAP|9,210|ReduceLROnPlateau|None|64|||
|Batch Norm + FC + GAP|9,186|None|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Dropout + GAP|9,178|None|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Transition + GAP|9,320|None|Rotation + Affine + Color Jitter|64|||
|Batch Norm + Transition + GAP|9,210|None|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + GAP|9,186|StepLR|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Dropout + GAP|9,178|StepLR|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Transition + GAP|9,320|StepLR|Rotation + Affine + Color Jitter|64|||
|Batch Norm + Transition + GAP|9,210|StepLR|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + GAP|9,186|ReduceLROnPlateau|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Dropout + GAP|9,178|ReduceLROnPlateau|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + Transition + GAP|9,320|ReduceLROnPlateau|Rotation + Affine + Color Jitter|64|||
|Batch Norm + Transition + GAP|9,210|ReduceLROnPlateau|Rotation + Affine + Color Jitter|64|||
|Batch Norm + FC + GAP|9,186|None|None|128|||
|Batch Norm + FC + Dropout + GAP|9,178|None|None|128|||
|Batch Norm + FC + Transition + GAP|9,320|None|None|128|||
|Batch Norm + Transition + GAP|9,210|None|None|128|||
|Batch Norm + FC + GAP|9,186|StepLR|None|128|||
|Batch Norm + FC + Dropout + GAP|9,178|StepLR|None|128|||
|Batch Norm + FC + Transition + GAP|9,320|StepLR|None|128|||
|Batch Norm + Transition + GAP|9,210|StepLR|None|128|||
|Batch Norm + FC + GAP|9,186|ReduceLROnPlateau|None|128|||
|Batch Norm + FC + Dropout + GAP|9,178|ReduceLROnPlateau|None|128|||
|Batch Norm + FC + Transition + GAP|9,320|ReduceLROnPlateau|None|128|||
|Batch Norm + Transition + GAP|9,210|ReduceLROnPlateau|None|128|||
|Batch Norm + FC + GAP|9,186|None|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Dropout + GAP|9,178|None|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Transition + GAP|9,320|None|Rotation + Affine + Color Jitter|128|||
|Batch Norm + Transition + GAP|9,210|None|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + GAP|9,186|StepLR|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Dropout + GAP|9,178|StepLR|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Transition + GAP|9,320|StepLR|Rotation + Affine + Color Jitter|128|||
|Batch Norm + Transition + GAP|9,210|StepLR|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + GAP|9,186|ReduceLROnPlateau|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Dropout + GAP|9,178|ReduceLROnPlateau|Rotation + Affine + Color Jitter|128|||
|Batch Norm + FC + Transition + GAP|9,320|ReduceLROnPlateau|Rotation + Affine + Color Jitter|128|||
|Batch Norm + Transition + GAP|9,210|ReduceLROnPlateau|Rotation + Affine + Color Jitter|128|||

## Results observed

## Our Findings
