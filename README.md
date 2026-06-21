# Wine Analytics Dimensionality Reduction with PCA

An unsupervised machine learning project showcasing feature decomposition and dimensionality reduction on multi-feature chemical profiles. This workspace demonstrates how standard scaling followed by orthogonal feature projection via Principal Component Analysis (PCA) simplifies dataset complexity while systematically measuring information preservation thresholds.

## Dataset Description
The pipeline utilizes the classic Wine recognition dataset fetched natively via Scikit-learn (`load_wine`). The data holds chemical analysis summaries of wines grown in a specific region in Italy, representing three distinct cultivars. Features describe physical and chemical dimensions, including alcohol content, malic acid, ash profiles, magnesium levels, total phenols, flavonoids, and color intensity.

## Methodology
1. **Data Ingestion & Integrity Check:** Parsed structural array matrices for features ($X$) and target classifications ($y$) from the raw data dictionary.
2. **Feature Standardization:** Deployed Scikit-learn's `StandardScaler` to perform z-score normalization across all features, mapping attributes to a common scale with a mean of 0 and variance of 1 to protect the PCA algorithm from scale-induced feature bias.
3. **PCA Dimensionality Reduction:** Initialized a 4-component transformation pipeline (`PCA(n_components=4)`) to fit and transform the scaled multi-dimensional feature space into orthogonal linear combinations.
4. **Variance Verification:** Evaluated structural retention constraints by computing the individual and cumulative explained variance ratios across the engineered component space.

## Results & Variance Analysis
Analyzing the principal component space outputted distinct information retention metrics across the decomposition tiers:
* **Principal Component 1 (PC1):** Accounted for **`36.20%`** of the global dataset variance.
* **Principal Component 2 (PC2):** Accounted for **`19.21%`** of the global dataset variance.
* **Principal Component 3 (PC3):** Accounted for **`11.12%`** of the global dataset variance.
* **Principal Component 4 (PC4):** Accounted for **`7.07%`** of the global dataset variance.

**Total Information Retention:** Compressing the high-dimensional feature matrix into just 4 principal components successfully preserved **`73.60%`** of the absolute cumulative variance of the original dataset, verifying a highly efficient structural reduction with minimal data loss.

## How to Run Locally
1. Clone this repository to your local machine environment.
2. Install standard scientific dependencies via pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
