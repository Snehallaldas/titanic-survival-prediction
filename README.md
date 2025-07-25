# 🚣️ Titanic Survival Prediction with MLOps

A full-stack Machine Learning project that predicts Titanic passenger survival using a trained Random Forest model, integrated with an ETL pipeline, feature store, drift detection, and monitoring.

---

## 🚀 Project Features

| Component         | Technology                         |
| ----------------- | ---------------------------------- |
| Model             | Random Forest (Accuracy: **0.86**) |
| ETL Pipeline      | Apache Airflow (Astro runtime)     |
| Feature Store     | Redis                              |
| Drift Detection   | Alibi-Detect (KSDrift)             |
| Monitoring        | Prometheus + Grafana               |
| Backend Framework | Flask                              |
| Frontend          | HTML + CSS                         |
| Metrics Endpoint  | `/metrics`                         |

---

## 🧠 Model Features

The model uses the following features to make survival predictions:

* `Age`
* `Fare`
* `Sex`
* `Pclass`
* `Embarked`
* `Familysize`
* `Isalone`
* `HasCabin`
* `Title`
* `Pclass_Fare`
* `Age_Fare`

---

## 🖥️ Web App Preview

The Flask web app allows users to enter feature values and receive a prediction:

* **"Survived"** or **"Did Not Survive"**

Form data is sent via POST request to `/predict` endpoint, and results are rendered using Jinja templates.

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/titanic-mlops.git
cd titanic-mlops
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python application.py
```

App runs at: `http://localhost:5000/`

### 5. Git Initialization & Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/titanic-mlops.git
git branch -M main
git push -u origin main
```


## 📈 Monitoring

* **Prometheus** scrapes metrics from `/metrics` endpoint.
* **Grafana** visualizes:

  * Prediction count
  * Drift count (data drift alerts)

---

## 🧪 Model Monitoring

This project integrates **Alibi-Detect** to identify data drift using the **KSDrift** detector. If drift is detected, it logs the event and increases a Prometheus counter.

---

## 🛠️ Tools Used

* Python 3.11+
* Flask
* Scikit-learn
* Pandas, NumPy
* Redis (as feature store)
* Alibi-detect
* Apache Airflow (ETL/Batch jobs)
* Prometheus & Grafana (Monitoring)

---


---

## 📜 License

This project is for educational purposes. Feel free to fork and extend it!

---

## 🙌 Acknowledgements

* Titanic dataset: [Kaggle](https://www.kaggle.com/c/titanic)
* Alibi Detect: [SeldonIO](https://github.com/SeldonIO/alibi-detect)
* Prometheus & Grafana for metrics monitoring

---

## ✨ Future Enhancements

* Dockerize the app for deployment
* CI/CD with GitHub Actions
* Retrain model on drift trigger
* Deploy via Render or AWS EC2

---
