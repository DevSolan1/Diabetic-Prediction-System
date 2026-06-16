# 🩺 Diabetes Prediction System

A Machine Learning web application that predicts the likelihood of diabetes based on patient health information. The application is built using **Python**, **Scikit-learn**, and **Streamlit** and provides an interactive user interface for making predictions.

---

## 📖 Project Overview

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help individuals seek medical attention sooner and improve health outcomes.

This project uses a **Logistic Regression Machine Learning model** trained on the Pima Indians Diabetes Dataset to predict whether a person is likely to have diabetes.

---

## ✨ Features

* Interactive and user-friendly Streamlit interface
* Data preprocessing and cleaning
* Missing value handling
* Feature scaling using StandardScaler
* Logistic Regression model for prediction
* Instant diabetes risk assessment
* Responsive and modern UI design

---

## 🛠 Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```text
Diabetic Prediction System/
│
├── diabete.py          # Main Streamlit application
├── diabetes.csv        # Dataset used for training
├── README.md           # Project documentation
└── requirements.txt    # Required libraries (optional)
```

---

## 📊 Dataset Information

The project uses the Pima Indians Diabetes Dataset containing medical information such as:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI (Body Mass Index)
* Diabetes Pedigree Function
* Age

Target Variable:

* Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Diabetic-Prediction-System.git
cd Diabetic-Prediction-System
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/Mac**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## ▶️ Run the Application

```bash
streamlit run diabete.py
```

After running the command, open the local URL displayed in the terminal.

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Handle Missing Values
3. Feature Preprocessing
4. Split Data into Training and Testing Sets
5. Scale Features
6. Train Logistic Regression Model
7. Generate Predictions
8. Display Results through Streamlit UI

---

## 📸 Application Preview

Add screenshots of your application here.

```text
screenshots/homepage.png
screenshots/prediction_result.png
```

---

## 🎯 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Machine Learning model training
* Classification using Logistic Regression
* Model deployment with Streamlit
* Building interactive data science applications

---

## 👨‍💻 Author

**Solan Abate Mekonin**

3rd Year Software Engineering Student

---

## ⭐ Support

If you found this project helpful, please consider giving it a star on GitHub.
