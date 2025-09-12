# 🩺 Diabetes Prediction Using Machine Learning

This project uses machine learning to predict whether a patient is diabetic based on medical attributes. It is built in Python and runs on Google Colab. The goal is to demonstrate how data science can be applied to healthcare diagnostics using classification models.

---

## 📊 Dataset Overview

We use the **Pima Indians Diabetes Dataset**, which contains diagnostic measurements for female patients of Pima Indian heritage. The dataset includes 768 samples and 8 medical features.

**Features:**
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Class variable (0 = Non-Diabetic, 1 = Diabetic)

---

## 🚀 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Visualize relationships between features and the target variable.
- Use boxplots and descriptive statistics to detect patterns.
- Identify which features are most correlated with diabetes.

### 2️⃣ Data Preprocessing
- Standardize features using `StandardScaler`.
- Split data into training and testing sets (80/20).
- Use pipelines to streamline preprocessing and modeling.

### 3️⃣ Model Building
We train and evaluate three classification models:
- **Logistic Regression**: Simple and interpretable.
- **Support Vector Machine (SVM)**: Effective for high-dimensional data.
- **Random Forest**: Ensemble method with strong performance.

Each model is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### 4️⃣ Hyperparameter Tuning
- Use `GridSearchCV` to optimize SVM parameters (`C`, `kernel`).
- Apply 5-fold cross-validation for robust evaluation.

### 5️⃣ Prediction Tool
- Input new patient data as a NumPy array.
- Output prediction: `Diabetic` or `Non-Diabetic`.

---

## 🛠 Technologies Used

- Python 3.x
- Google Colab
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 📦 How to Run the Project

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially.
3. Modify the `new_patient` array to test predictions on custom data.

```python
new_patient = [2, 120, 70, 25, 100, 30.5, 0.5, 35]


Model Comparison:
- Logistic Regression Accuracy: 0.78
- SVM Accuracy: 0.81
- Random Forest Accuracy: 0.84

Best Model: SVM with RBF kernel
Prediction for new patient: Diabetic


├── diabetes_prediction.ipynb
├── README.md
├── requirements.txt
└── data/
    └── pima_diabetes.csv