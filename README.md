# Wine Dataset Clustering — Unsupervised Machine Learning

## 📌 Project Overview

This project explores unsupervised machine learning techniques for clustering the Wine dataset.

Three different clustering algorithms were implemented and compared:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

PCA (Principal Component Analysis) was also used for dimensionality reduction and visualization.

## 📊 Dataset

The Wine dataset is a built-in dataset provided by Scikit-learn.

It contains chemical measurements of wines with multiple numerical features.

## 🔧 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🧠 Techniques Used

### 1. Data Preprocessing

StandardScaler was used to standardize the features before applying clustering algorithms.

### 2. PCA

Principal Component Analysis (PCA) was used to reduce the dimensionality of the dataset and visualize the clusters in lower dimensions.

### 3. K-Means Clustering

K-Means was applied to divide the data into different clusters.

The appropriate number of clusters was investigated using clustering evaluation techniques.

### 4. Hierarchical Clustering

Hierarchical clustering was implemented to identify groups within the dataset.

### 5. DBSCAN

DBSCAN was used as a density-based clustering algorithm and compared with the other clustering approaches.

## 📈 Model Evaluation

The clustering algorithms were evaluated using clustering evaluation metrics such as:

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

The results were compared to understand which clustering algorithm performed better on the dataset.

## 📌 Conclusion

This project provided practical experience with multiple unsupervised machine learning algorithms and demonstrated how different clustering techniques can produce different grouping patterns on the same dataset.

The project also demonstrated the use of PCA for dimensionality reduction and visualization.
