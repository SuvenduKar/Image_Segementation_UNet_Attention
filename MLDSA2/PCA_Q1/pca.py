import os
import cv2
import numpy as np
import pandas as pd

#Define the function for PCA
def pca(X, n_components):
    # Calculate mean of the data
    mean = np.mean(X, axis=0)
    
    # Center the data
    X_centered = X - mean
    
    # Compute covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select top n_components eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    # Project the data onto the selected eigenvectors
    X_pca = np.dot(X_centered, top_eigenvectors)
    
    return X_pca, top_eigenvectors


# Load images
folder = "C:/Users/sk731/OneDrive/Desktop/MLDSA2/train/my_train_img/"
images = []
for i in range(919):#Total number of images in the folder
    img_path = os.path.join(folder, f"img_{i}.png")
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, (16, 16))#Resizing the image to 16*16*3
    images.append(resized_image.flatten())
X = np.array(images)

# Apply PCA to reduce dimensionality to 10 components
X_pca, _ = pca(X, 2)

# Load CSV file
csv_file = "C:/Users/sk731/OneDrive/Desktop/MLDSA2/train.csv"
df = pd.read_csv(csv_file)

# Drop "id" column from CSV
df.drop(columns=["id"], inplace=True)

# Add 2 principal components to DataFrame
for i in range(2):
    df[f"PC{i+1}"] = X_pca[:, i]

# Save updated CSV file
df.to_csv("C:/Users/sk731/OneDrive/Desktop/MLDSA2/my_train_pca_2.csv", index=False)
print("Updated CSV file saved successfully")