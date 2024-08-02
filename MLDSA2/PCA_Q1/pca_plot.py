import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("C:/Users/sk731/OneDrive/Desktop/MLDSA2/PCA_Q1/my_train_pca_2.csv")

# Separate points based on their label
label_0 = data[data['binary_pred'] == 0]
label_1 = data[data['binary_pred'] == 1]

# Plot the points
plt.scatter(label_0['PC1'], label_0['PC2'], color='green', label='Label 0')
plt.scatter(label_1['PC1'], label_1['PC2'], color='red', label='Label 1')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Save the plot as an image
plt.savefig('C:/Users/sk731/OneDrive/Desktop/MLDSA2/PCA_Q1/pca_plot.png')

# Show the plot
plt.show()
