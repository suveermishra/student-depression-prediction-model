# student-depression-prediction-model
End-to-end ML web app that predicts early-stage diabetes risk from clinical and lifestyle indicators. Flask-based application delivering real-time predictions using a Random Forest classifier (AUC ~0.94), selected after benchmarking against 4 other models.


Student Depression Risk Predictor

A complete machine learning pipeline that estimates a student's risk of depression using academic, lifestyle, and demographic indicators, delivered through a real-time Flask web application.

Overview

A student's mental wellbeing is influenced by a mix of factors — academic workload, sleep patterns, eating habits, financial pressures, and more. This project builds a full ML workflow around that idea:

Data preprocessing — cleaning and encoding a mental health dataset collected from students
Model training — building and benchmarking four different classification algorithms
Model selection — Logistic Regression was chosen for production use due to its solid AUC score (~0.92) and ease of interpretation
Deployment — a Flask-based web app that serves live predictions through a straightforward form interface
Features
An input form capturing relevant details: age, gender, city, academic pressure, CGPA, sleep duration, dietary habits, degree program, work/study hours, financial stress, and family history of mental illness
Instant depression risk prediction along with a probability estimate
Robust handling of previously unseen categories (such as new city names) through safe label encoding, so the app doesn't break on unfamiliar input
Backend validation to keep inputs within realistic bounds (age, CGPA, work/study hours)
Tech Stack
Layer	Technology
Backend	Python, Flask
Machine Learning	scikit-learn (Logistic Regression), pandas, NumPy, joblib
Frontend	HTML, CSS
Project Structure
├── app.py                          # Flask application and prediction logic
├── depression_model.pkl            # Trained Logistic Regression model
├── student_depression_dataset.csv  # Training dataset
├── static/
│   └── style.css                   # Styling
└── templates/
    └── index.html                  # Web form UI
Getting Started
Prerequisites

Python 3.8+

Installation
bash
git clone https://github.com/your-username/student-depression-prediction-model.git
cd student-depression-prediction-model
pip install flask pandas numpy scikit-learn joblib
Running the app
bash
python app.py

Open http://localhost:5000 in your browser to use it.

Model Details

During development, four different classification algorithms were trained and compared. Logistic Regression was picked as the final model because of:

An AUC of roughly 0.92 on held-out test data
Its interpretability — a key consideration for a sensitive domain like mental health screening, where understanding the reasoning behind a prediction matters just as much as the prediction itself
Disclaimer

This project was built as an academic minor project for learning purposes only, and is not intended as a diagnostic or clinical tool. Its predictions are derived from patterns in a public dataset, not from individual clinical evaluation. Anyone experiencing mental health difficulties is encouraged to consult a qualified professional or reach out to a local support helpline.

Team
Suveer Mishra (EC23B023)
Abhinabo Acharjee (EC23B001)

Supervised by Dr. Yang Saring, NIT Arunachal Pradesh
