# Credit-Risk-Prediction-Model
# Credit Risk Scoring & Predictive Analytics Pipeline

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
* Analyzed the distribution of the target variable (`loan_status`) to identify the class imbalance between paid loans and defaults.
* Generated a correlation matrix to identify high-risk financial drivers. 
    * *Insight:* Loan interest rates (`loan_int_rate`) and the loan-to-income percentage (`loan_percent_income`) showed the strongest positive correlation with default probability.

![Correlation Matrix](correlation_matrix.png)

### 2. Data Cleaning & Feature Engineering
* **Imputation:** Handled missing data in critical columns (`person_emp_length` and `loan_int_rate`) by filling them with the median values to prevent outlier distortion.
* **Encoding:** Converted categorical text variables (e.g., loan intent, home ownership) into numeric format using One-Hot Encoding (`pd.get_dummies`).
* **Scaling:** Applied `StandardScaler` to ensure features with large numerical ranges (like income) did not disproportionately influence the model compared to smaller-range features (like age).

### 3. Machine Learning Model Formulation
* **Algorithm:** Implemented a Random Forest Classifier (`n_estimators=100`).
* **Imbalance Handling:** Utilized `class_weight='balanced'` during model initialization to penalize misclassifications of the minority class (defaulters), forcing the model to prioritize risk detection over pure accuracy.

### 4. Business Impact & Evaluation
Instead of relying solely on baseline accuracy, the model was evaluated on metrics relevant to institutional risk officers:
* **ROC-AUC Score:** Calculated to measure the model's capability to distinguish between safe and high-risk applicants.
* **Risk Avoidance Metrics:** Extracted the Confusion Matrix to calculate the exact number of True Positives (defaulters correctly blocked) versus False Negatives (defaulters incorrectly approved/risk leakage).

## 🚀 How to Run the Notebook
The full data pipeline and model execution can be viewed and run interactively here:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) *(Note: Add your specific shareable Colab link here)*
