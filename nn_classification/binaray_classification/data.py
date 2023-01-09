from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Make 1000 samples 
n_samples = 1000

# Generate two circles with different coloured dots. 
# red (0) or blue (1).
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values
# print(f"First 5 X features:\n{X[:5]}")
# print(f"\nFirst 5 y labels:\n{y[:5]}")

# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
# print(f'==>> circles.head(10):\n{circles.head(10)}')

# Check different labels
# print(f'==>> circles.label.value_counts():\n{circles.label.value_counts()}')

# Visualize with a plot
# plt.scatter(x=X[:, 0], 
#             y=X[:, 1], 
#             c=y, 
#             cmap=plt.cm.RdYlBu)
# plt.show()

# Check the shapes of our features and labels
# print("==>> X.shape: ", X.shape)
# print("==>> y.shape: ", y.shape)

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
# print("==>> X[:5]: ", X[:5])
# print("==>> y[:5]: ", y[:5])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

# print("==>> len(X_train): ", len(X_train))
# print("==>> len(X_test): ", len(X_test))
# print("==>> len(y_train): ", len(y_train))
# print("==>> len(y_test): ", len(y_test))