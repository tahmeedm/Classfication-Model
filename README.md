# PCA and Clustering Analysis of Wheat Kernels

## Project Overview

This project focuses on analyzing the geometric properties of wheat kernels from three varieties: Kama, Rosa, and Canadian, utilizing Principal Component Analysis (PCA) and K-Means clustering. The primary goal is to enhance dimensionality reduction and unsupervised learning techniques. Additionally, Support Vector Machines (SVM) are employed as the main model for classification due to their robustness with high-dimensional data. The project is implemented using Python with libraries such as Scikit-learn and Matplotlib.

## Dataset Description

The dataset pertains to a study on the geometrical properties of wheat kernels from three different varieties. Here are some key characteristics:
- **Dataset Characteristics**: Multivariate, real-valued, suitable for classification and clustering tasks in biology.
- **Measurement Technique**: Soft X-ray technique for high-quality, non-destructive visualization of internal kernel structures.
- **Geometric Parameters**: Seven parameters measured for each kernel: area (A), perimeter (P), compactness (C = 4πA/P²), length, width, asymmetry coefficient, and length of kernel groove.
- **Research Purpose**: Facilitates analysis of features in X-ray images of wheat kernels for statistical and machine learning tasks.

## Project Structure

### 1. Data Loading and Preprocessing

1. **Download and Load Data**:
    ```python
    !wget -N "https://archive.ics.uci.edu/static/public/236/seeds.zip"
    !unzip -o seeds.zip seeds_dataset.txt
    !rm -r seeds.zip
    import pandas as pd
    data = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None)
    data.columns = ['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel', 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Type']
    display(data)
    ```

2. **Split Data**:
    ```python
    from sklearn.model_selection import train_test_split
    X = data.drop('Type', axis=1)
    y = data['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ```

### 2. PCA and Visualization

1. **Preprocessing and Scaling**:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

2. **PCA Implementation**:
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    ```

3. **2D Scatter Plot**:
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    classes = np.unique(y_train)
    markers = ['o', 's', '^']
    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for class_label, marker, color in zip(classes, markers, colors):
        ax[0].scatter(X_train_pca[y_train == class_label, 0], X_train_pca[y_train == class_label, 1], marker=marker, color=color, label=f'Class {class_label}', alpha=0.5)
    ax[0].set_title('Training Set After PCA')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].legend()
    for class_label, marker, color in zip(classes, markers, colors):
        ax[1].scatter(X_test_pca[y_test == class_label, 0], X_test_pca[y_test == class_label, 1], marker=marker, color=color, label=f'Class {class_label}', alpha=0.5)
    ax[1].set_title('Testing Set After PCA')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].legend()
    plt.show()
    ```

### 3. SVM Classification

1. **Model Selection and Hyperparameter Tuning**:
    ```python
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svm__degree': [2, 3, 4]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-validation Accuracy:", grid_search.best_score_)
    optimized_model = grid_search.best_estimator_
    ```

2. **Model Performance with PCA**:
    ```python
    grid_search_pca = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_pca.fit(X_train_pca, y_train)
    print("Best Hyperparameters with PCA:", grid_search_pca.best_params_)
    print("Best Cross-validation Accuracy with PCA:", grid_search_pca.best_score_)
    print("Best Cross-validation Accuracy without PCA:", grid_search.best_score_)
    ```

### 4. Clustering with K-Means

1. **K-Means Clustering**:
    ```python
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer
    clustering_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(random_state=42))
    ])
    visualizer = KElbowVisualizer(clustering_pipeline.named_steps['kmeans'], k=(2,10), metric='calinski_harabasz')
    visualizer.fit(X)
    visualizer.show()
    optimal_k = visualizer.elbow_value_
    clustering_pipeline.set_params(kmeans=KMeans(n_clusters=optimal_k, random_state=42))
    clustering_pipeline.fit(X)
    ```

## Results

- **SVM Model Performance**: Achieved a cross-validation accuracy of 96.4% with hyperparameter tuning.
- **Dimensionality Reduction**: PCA helped visualize the data in two dimensions, though the SVM performed better with the original higher-dimensional data.
- **Clustering**: Optimal number of clusters determined to be 3 using the Calinski-Harabasz metric.

## Conclusion

This project demonstrates the effectiveness of PCA and K-Means clustering for dimensionality reduction and unsupervised learning, respectively. The SVM model proved highly effective for classification tasks, achieving excellent accuracy through careful preprocessing and hyperparameter optimization.

## References

- UCI Machine Learning Repository: [Seeds Dataset](https://archive.ics.uci.edu/dataset/236/seeds)
- Scikit-learn Documentation
- Matplotlib Documentation
- Yellowbrick Documentation
