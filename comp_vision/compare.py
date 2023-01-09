import pandas as pd
from cnn_ex import model_2_results, total_train_time_model_2
from gpu_model import model_1_results, total_train_time_model_1
from predict import model_0_results, total_train_time_model_0
import matplotlib.pyplot as plt

compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
print("==>> compare_results")
print(compare_results)

# Add training times to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
print("==>> compare_results")
print(compare_results)

# Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.show()