# Credit-Risk-Prediction-Model

## 📌 Project Overview
This project develops an end-to-end Machine Learning classifier to predict the likelihood of financial loan defaults. Utilizing historical credit applicant data, the pipeline cleans features, handles missing variables, balances skewed default classes, and trains a predictive model to optimize a financial institution's risk management strategy.

## 👤 Author
**Rihab Junaid Basheer Ahmed** *MSc Business Analytics and Data Science | Politecnico di Milano*

## 🛠️ Tech Stack & Methodology
* **Environment:** Python (Google Colab)
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Model Engine:** Random Forest Classifier 

## ⚙️ What I Did: The Pipeline Step-by-Step

### 1. Exploratory Data Analysis (EDA)
* Analyzed the distribution of the target variable (`loan_status`) to identify the baseline class imbalance between fully paid loans and defaults.
* Generated a correlation matrix for all numerical applicant metrics. 
    * *Key Insight:* `loan_percent_income` and `loan_int_rate` demonstrated the strongest positive correlations with loan defaults, making them primary risk drivers.

### 2. Data Cleaning & Preprocessing
* **Imputation:** Addressed missing data in the `person_emp_length` and `loan_int_rate` columns by replacing nulls with the median values.
* **Feature Encoding:** Transformed categorical text features (such as loan intent) into numeric format using One-Hot Encoding (`drop_first=True` to avoid multicollinearity).
* **Data Splitting:** Executed an 80/20 train-test split, strictly stratifying the target variable to maintain accurate default proportions across both sets.
* **Feature Scaling:** Applied `StandardScaler` to normalize the numerical ranges so large metrics (like annual income) did not overshadow smaller metrics (like age or employment length).

### 3. Model Training
* **Algorithm:** Implemented a Random Forest Classifier (`n_estimators=100`, `random_state=42`).
* **Imbalance Handling:** Applied the `class_weight='balanced'` parameter, forcing the algorithm to mathematically prioritize the detection of the minority class (actual defaulters) over simply maximizing baseline accuracy.

### 4. Model Performance & Business Metrics
The model was tested on a holdout set of 6,517 applicants.

* **Overall Accuracy:** 93%
* **Defaulter Precision (Class 1):** 96% — When the model flags a high-risk applicant, it is overwhelmingly correct, minimizing false alarms that would turn away good customers.
* **Defaulter Recall (Class 1):** 72% — The model successfully identified and blocked 72% of all actual defaulting loans.
* **F1-Score (Class 1):** 0.82
* **Safe Applicant Recall (Class 0):** 99% — The model accurately cleared 99% of reliable borrowers, ensuring the financial institution does not lose profitable loan volume.

## 🚀 How to Run the Notebook
The full data pipeline and model execution can be viewed and run interactively https://colab.research.google.com/drive/1TxMSKbobHUrPpCXvvaCTn46YgpARDUiW#scrollTo=-1-hRnOD3fZ3
[![Open In Colab](https://colab.research.gooarch.google.com/) *(Note: Add your specific shareable Colab link here)*
