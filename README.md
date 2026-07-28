# 🍃 Guava Disease Detection using Deep Learning

A web-based Deep Learning application that detects diseases in guava leaves from uploaded images. The project demonstrates the complete lifecycle of an AI application—from model training to deployment—using modern Machine Learning, DevOps, and Cloud technologies.

---

## Overview

This application allows users to upload an image of a guava leaf through a web interface. A trained TensorFlow model analyzes the image and predicts the corresponding disease along with a confidence score.

Beyond model development, this project focuses on deploying a Machine Learning model in a production-like environment using Docker, Jenkins, GitHub Webhooks, and AWS EC2.

---

## Features

- Upload guava leaf images through a web interface
- Predict guava leaf diseases using a Deep Learning model
- Display prediction confidence
- Simple and responsive user interface
- Dockerized application for consistent deployment
- Automated CI/CD pipeline using Jenkins
- Automatic deployment triggered through GitHub Webhooks
- Hosted on AWS EC2

---

## Technologies Used

### Machine Learning
- TensorFlow
- Keras
- NumPy
- OpenCV

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS

### DevOps & Cloud
- Docker
- Jenkins
- GitHub Webhooks
- AWS EC2

### Version Control
- Git
- GitHub

---

## Project Workflow

```
User Uploads Image
        │
        ▼
 Flask Web Application
        │
        ▼
TensorFlow Deep Learning Model
        │
        ▼
 Disease Prediction
        │
        ▼
 Prediction Displayed
```

---

## CI/CD Workflow

```
Developer
    │
    ▼
Git Commit
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Webhook
    │
    ▼
Jenkins Pipeline
    │
    ▼
Docker Image Build
    │
    ▼
Docker Container
    │
    ▼
AWS EC2 Deployment
```

---

## Project Structure

```
guava_disease_detection/

├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
├── saved_models/
├── static/
├── templates/
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Dhrupu/guava_disease_detection.git
cd guava_disease_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Docker

Build the Docker image

```bash
docker build -t guava-app .
```

Run the container

```bash
docker run -d -p 5000:5000 --name guava-container guava-app
```

---

## Deployment

The application is deployed on an AWS EC2 instance using Docker containers.

A Jenkins pipeline automatically builds and deploys the latest version whenever changes are pushed to the configured GitHub branch through GitHub Webhooks.

---

## Future Improvements

- Interactive dashboard
- Disease descriptions and treatment suggestions
- Prediction history
- User authentication
- Mobile responsive interface
- Analytics dashboard
- Model explainability using Grad-CAM
- HTTPS and custom domain
- Monitoring with Prometheus and Grafana

---

## Author

**C R Dhrupadh**

Bachelor of Vocation in Data Science

GitHub: https://github.com/Dhrupu

Project Repository:
https://github.com/Dhrupu/guava_disease_detection

---

## License

This project was developed for educational and learning purposes.