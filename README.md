# Predictive Healthcare Analytics: Diabetes & Heart Disease Prediction

https://predictive-healthcare-eykrhpkgebikswuhw9cvko.streamlit.app/

A comprehensive end-to-end Machine Learning pipeline implemented in Google Colab that addresses both supervised classification tasks and unsupervised cluster analysis to assess clinical health risks. The pipeline trains, cross-validates, and compares multiple predictive models to isolate high-risk patients based on diagnostic criteria.

---

## 🚀 Key Features

* **Dual-Disease Analytical Pipeline:** Dedicated independent tracking modules for both Diabetes (PIMA Indians Dataset) and Cardiovascular Pathology (UCI Cleveland Dataset).
* **Robust Preprocessing Infrastructure:** Auto-replaces structural biological anomalies (e.g., impossible zero-readings for metrics like blood pressure/insulin) with robust statistical medians before scaling features using `StandardScaler`.
* **Multi-Classifier Benchmarking:** Compares 6 distinct machine learning paradigms natively:
  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Ensemble
  * Support Vector Machine (RBF Kernel)
  * Multilayer Perceptron (MLP) Neural Network
  * Gaussian Naive Bayes
* **Unsupervised Risk Profiling:** Incorporates K-Means Clustering combined with automated Principal Component Analysis (PCA) dimensionality reduction to trace population risk patterns without target variables.
* **Production Deployment Ready:** Serializes trained optimal model parameters and associated data scalers to `.pkl` formats via `joblib`, ready for a live Streamlit dashboard deployment.

---

## 📦 System Architecture & Stack

* **Language Platform:** Python 3.x
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`
* **Visualization Engine:** `matplotlib`, `seaborn`
* **Target Interface Framework:** `streamlit`, `pyngrok` (Pre-configured setup hooks)

---

## 📊 Dataset Metadata

### 1. Diabetes Dataset (PIMA Indians)
* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases
* **Instances:** 768 entries, 9 attributes
* **Predictive Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)
* **Key Predictive Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### 2. Heart Disease Dataset (UCI Cleveland)
* **Source:** Processed Cleveland Clinic Foundation Dataset
* **Instances:** 303 entries, 14 attributes
* **Predictive Target:** `target` (0 = Healthy, 1 = Diagnosed Heart Disease)
* **Key Predictive Features:** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs`, `restecg`, `thalach` (max heart rate), `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## 🛠️ Execution & Project Workflow

### Step 1: Initialization
Fetches required modules and updates remote computing environments for hosting packages (`Streamlit`, `pyngrok`).

### Step 2: Ingestion & Structural Cleaning
Pulls raw remote streams natively via public URLs. To prevent standard classification errors, multi-tiered heart conditions (stages 1-4) are cleanly binarized into explicit status arrays (`0` vs `1`).

### Step 3: Exploratory Data Analysis (EDA)
Generates comprehensive visual metrics for each operational run:
* Bar-charts graphing baseline balance profiles across positive/negative diagnoses.
* High-definition multi-variable Pearson correlation matrices (`sns.heatmap`).
* Distribution/outlier evaluations.

### Step 4: Missing Value Neutralization & Feature Engineering
Replaces invalid zero records in biological fields with data-imputed statistical medians (`SimpleImputer`). Splitting uses a stratified `80/20` format to guarantee correct representation in train and test splits, followed by Z-score standardization.

### Step 5 & 6: Grid Benchmarking & Visualization Metrics
Models run parallel scoring loops across test slices. The script plots performance tracking arrays:
* Accuracy bar metrics across classifiers.
* Custom Multi-Metric charts (`Accuracy`, `Precision`, `Recall`, `F1`).
* **Receiver Operating Characteristic (ROC)** curve charts accompanied by calculating area scores (AUC).
* Confusion matrix thermal mappings targeting the highest performing iteration.

### Step 7: K-Means Cluster Clustering
Executes an Elbow-Inertia optimization pass between $k=[2,9]$. Reduces scaled features down to 2 components via PCA to provide clean scatter-maps indicating target centers.

### Step 8 & 9: Model Export & Interactive Testing Functions
Exports operational assets (`.pkl`) and defines real-time validation wrappers (`predict_diabetes()`, `predict_heart_disease()`) returning complete data assessments:
* **Prediction Status:** Binary Diagnostic String Output
* **Risk Probability:** Exact Confidence Percentages
* **Risk Level Assignment:** Formatted Threat Threshold Scaling (`Low` $\leq$ 40%, `Medium` 41%-70%, `High` $>$ 70%)

---

## 💻 Sample Validation Log Example

```text
  Diabetes Prediction:
   Input: [6, 148, 72, 35, 0, 33.6, 0.627, 50]
   Prediction: Diabetic
   Risk Probability: 76.42%
   Risk Level: High

❤️ Heart Disease Prediction:
   Input: [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
   Prediction: Heart Disease
   Risk Probability: 88.19%
   Risk Level: High
