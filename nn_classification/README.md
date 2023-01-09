# Architecture of a classification neural network

| Hyperparameter       | Binary Classification  | Multi-class classification |
| -------------------- | ---------------------- | ------------------------- |
| Input layer shape    | Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction)  | Same as binary classification |
| Hidden layer(s)      | Problem specific, minimum = 1, maximum = unlimited  | Same as binary classification |
| Neurons per hidden layer   | Problem specific, generally 10 to 512  | Same as binary classification |
| Output layer shape   | 1 (one class or the other)  | 1 per class (e.g. 3 for food, person or dog photo) |
| Hidden layer activation   | Usually ReLU (rectified linear unit) but can be many others  | Same as binary classification |
| Output activation   | Sigmoid (torch.sigmoid in PyTorch)  | Softmax (torch.softmax in PyTorch) |
| Loss function   | Binary crossentropy (torch.nn.BCELoss in PyTorch)  | Cross entropy (torch.nn.CrossEntropyLoss in PyTorch) |
| Optimizer   | SGD (stochastic gradient descent), Adam (see torch.optim for more options)  | Same as binary classification |

## Loss Function & Optimizer

| Loss function/Optimizer | Problem type | PyTorch Code |
| --- | --- | --- |
| Stochastic Gradient Descent (SGD) optimizer | Classification, regression, many others. | `torch.optim.SGD()` |
| Adam Optimizer | Classification, regression, many others. | `torch.optim.Adam()` |
| Binary cross entropy loss | Binary classification | `torch.nn.BCELossWithLogits` or `torch.nn.BCELoss` |
| Cross entropy loss | Mutli-class classification | `torch.nn.CrossEntropyLoss` |
| Mean absolute error (MAE) or L1 Loss | Regression | `torch.nn.L1Loss` |
| Mean squared error (MSE) or L2 Loss | Regression | `torch.nn.MSELoss` |

## Improving a Model (from a model perspective)

| Model improvement technique | What does it do? |
| --- | --- |
| Add more layers | Each layer potentially increases the learning capabilities of the model with each layer being able to learn some kind of new pattern in the data, more layers is often referred to as making your neural network deeper. |
| Add more hidden units | Similar to the above, more hidden units per layer means a potential increase in learning capabilities of the model, more hidden units is often referred to as making your neural network wider. |
| Fitting for longer (more epochs) | Your model might learn more if it had more opportunities to look at the data. |
| Changing the activation functions | Some data just can't be fit with only straight lines (like what we've seen), using non-linear activation functions can help with this (hint, hint). |
| Change the learning rate | Less model specific, but still related, the learning rate of the optimizer decides how much a model should change its parameters each step, too much and the model overcorrects, too little and it doesn't learn enough. |
| Change the loss function | Again, less model specific but still important, different problems require different loss functions. For example, a binary cross entropy loss function won't work with a multi-class classification problem. |
| Use transfer learning | Take a pretrained model from a problem domain similar to yours and adjust it to your own problem. We cover transfer learning in notebook 06. |

## Classification evaluation metrics

| Metric name/Evaluation method | Defintion | Code |
| --- | --- | --- |
| Accuracy | Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct. | `torchmetrics.Accuracy()` or `sklearn.metrics.accuracy_score()` |
| Precision | Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0). | `torchmetrics.Precision()` or `sklearn.metrics.pre
