import matplotlib.pyplot as plt
from data import X_train, X_test, y_train, y_test
from train import model_0
from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

# The model is currently trying to split the red and blue dots using a straight line...
# That explains the 50% accuracy. Since our data is circular, drawing a straight line can at best cut it down the middle.
# The model is underfitting, meaning it's not learning predictive patterns from the data.
