# 💎 Diamond Dynamics: Price Prediction and Market Segmentation
---

## 📁 Project Structure

```
diamond_dynamics/
├── diamond_dynamics.ipynb   ← Main Jupyter Notebook (run this first)
├── app.py                   ← Streamlit Web Application
├── requirements.txt         ← Python dependencies
└── README.md
```

After running the notebook, these files are generated automatically:
```
├── best_regression_model.pkl
├── ann_model.keras
├── kmeans_model.pkl
├── scaler.pkl
├── scaler_cluster.pkl
├── ordinal_encoder.pkl
├── cluster_names.pkl
└── model_meta.json
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Notebook
Open `diamond_dynamics.ipynb` in Jupyter and run all cells top-to-bottom.
This will train all models and save `.pkl` files automatically.

### Step 3 — Launch Streamlit App
```bash
streamlit run app.py
```
Open browser at http://localhost:8501

---

## 📊 What the Notebook Covers

| Section | Details |
|---------|---------|
| Data Loading | Seaborn diamonds dataset (53,940 × 10) |
| Preprocessing | Null handling, zero-value imputation |
| Outlier Handling | IQR method on carat, price, x, y, z |
| Skewness Handling | log1p transformation |
| EDA | Distributions, boxplots, heatmap, pairplot |
| Feature Engineering | Volume, Price/Carat, Dimension Ratio, INR conversion |
| Encoding | OrdinalEncoder for cut, color, clarity |
| Feature Selection | VIF analysis |
| Regression | Linear, DT, RF, XGBoost, KNN + ANN |
| Clustering | K-Means (Elbow + Silhouette), k=3 |
| PCA | 2D and 3D visualizations |
| Model Saving | All models saved as .pkl / .keras |

---

## 🤖 Models Used

### Regression (Price Prediction)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor ← typically best (R²≈0.98)
- K-Nearest Neighbors
- ANN (TensorFlow/Keras) — 3 hidden layers, BatchNorm, Dropout

### Clustering (Market Segmentation)
- K-Means (k=3, determined by Elbow + Silhouette methods)
- PCA for 2D/3D cluster visualization

### Cluster Labels
| Cluster | Description |
|---------|-------------|
| 💎 Premium Heavy Diamonds | High carat, high price, premium grade |
| 🔷 Mid-range Balanced Diamonds | Medium carat and price |
| 💠 Affordable Small Diamonds | Low carat, budget-friendly |

---

## 📱 Streamlit App Features
- **Price Prediction Module:** Input diamond attributes → Predicted price in INR
- **Market Segmentation Module:** Identify which market cluster a diamond belongs to
- **Interactive UI** with real-time outputs and visual insights

---

## 📦 Dataset
- Source: `seaborn.load_dataset('diamonds')` — same as the Kaggle diamonds dataset
- Shape: 53,940 rows × 10 columns
- Target: `price` (USD) → converted to INR

- 
