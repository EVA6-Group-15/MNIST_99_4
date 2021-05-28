# Train models on MNIST Dataset to achieve 99.4 validation accuracy

## Goal To achieve

The activity that was performed using MNIST data was to achieve a validation accuracy of atleast 99.4% within 20 epochs of training the model and ensuring the number of parameters are less than 20,000. The purpose of this activity was to understand how we can consider different types of network architectures, considering different loss functions can help us achieve the desired result.

Below we have explained in detail on how we achieved the goal and what was our findings as much detailed possible for us.

## Network Architectures Used

To test out what was the best approach, we used 6 different network architectures. The architecture diagram along with the model summary are listed below:

### Network - Vanilla

Vanilla Network: **Architecture Diagram**

![n1arch](./../images/network_1.png)

Vanilla Network: **Model parameters**

![n1param](./../images/network1_parameters.png)

### Network - With Dropout

Network With Dropout: **Architecture Diagram**

![n4arch](./../images/network_1.png)

Network With Dropout: **Model parameters**

![n4param](./../images/network1_parameters.png)

### Network - With GAP and Transition Layer

Network With GAP and Transition Layer: **Architecture Diagram**

![n2arch](./../images/network_1.png)

Network With GAP and Transition Layer: **Model parameters**

![n2param](./../images/network1_parameters.png)

### Network - With GAP and Transition Layer without FC

Network GAP and Transition Layer without FC: **Architecture Diagram**

![n3arch](./../images/network_1.png)

Network GAP and Transition Layer without FC: **Model parameters**

![n3param](./../images/network1_parameters.png)

### Why particular changes were made in the neural Architecture

## Training the models

## Results observed

## Our Findings
