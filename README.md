# Telecom-Customer-Churn-Prediction

This project develops a complete machine learning workflow to predict customer churn for a telecommunications company. It demonstrates end-to-end data preparation, exploratory analysis, feature engineering, model development, and deployment readiness.

---

## 📊 Project Overview and Data Handling
- Built an end-to-end **Machine Learning model** to predict churn.  
- Dataset: 3,333 customer records with 21 attributes.  
- Identified a highly **imbalanced binary classification problem** (85% non-churn, 14.5% churn).  
- Corrected feature data types (e.g., `Area Code` → categorical).  
- Applied **ADASYN (Adaptive Synthetic Sampling)** to address severe class imbalance, outperforming SMOTE in complex data regions.  

---

## 🔍 Exploratory Data Analysis (EDA) & Feature Engineering
- Conducted **EDA** using violin plots and correlation matrices to identify churn predictors.  
- Key signals:  
  - **Total Day Minutes/Charge** and **Customer Service Calls**.  
  - Customers with an **International Plan** churned 4x more frequently.  
- Applied transformations:  
  - Binning (`Voicemail Messages`, `Customer Service Calls`).  
  - Log(1+X) transformation for skewed variables (e.g., `Total International Calls`).  
- Prevented **data leakage** by performing a stratified 80/20 train-test split *before* scaling and encoding.  
- Encoding strategies:  
  - **Frequency Encoding** for low-cardinality weak predictors (e.g., `Area Code`).  
  - **Target Encoding** for high-cardinality strong predictors (e.g., `State`).  
- Addressed **multicollinearity** by removing redundant charge columns (100% correlated with minutes).  
- Normalized numeric features using **RobustScaler** to reduce outlier influence.  

---

## 🤖 Model Development
- Tested multiple classifiers: **Logistic Regression, KNN, LightGBM**.  
- Used **5-fold cross-validation** to avoid overfitting.  
- Evaluation Metrics: Precision, Recall, F1 Score, ROC AUC (prioritized over Accuracy).  
- Selected **LightGBM Classifier**:  
  - Achieved **~88% ROC AUC** with a strong F1 Score.  
- Hyperparameter Tuning:  
  - Applied **RandomizedSearchCV** focusing on **F1 Score** to balance precision and recall.  
- Validated predictions via **Confusion Matrix** to confirm performance on churn vs. non-churn.  

---

## 🚀 Production Pipeline
- Designed a reusable **Machine Learning Pipeline** with:  
  - `ColumnTransformer` + `FunctionTransformer` for preprocessing.  
  - Automated encoding, scaling, and feature transformation.  
- Serialized the trained **LightGBM model** into a `.pkl` file for deployment.  
- Enables inference on new customer data without retraining.  

---

## 📂 Repository Structure
