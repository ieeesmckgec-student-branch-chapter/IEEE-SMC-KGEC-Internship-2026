# Explainable and Generative AI-Based Cardiovascular Risk Prediction for Young Adults

An AI-powered cardiovascular risk prediction system designed to estimate cardiovascular risk in young adults using machine learning, explainable AI, and generative AI.

The system combines multiple machine learning models to generate a risk prediction, uses SHAP to explain the important factors influencing the prediction, and uses Generative AI to convert the technical prediction into a human-readable report.

---

## 📌 Project Overview

Cardiovascular disease is often associated with older populations, but risk factors can begin developing much earlier in life. Early identification of cardiovascular risk may help individuals understand potentially important lifestyle and health-related factors.

This project focuses on cardiovascular risk prediction for young adults using an ensemble machine learning approach.

The system provides:

- Cardiovascular risk prediction using an ensemble ML model
- Predicted risk probability
- Risk category classification
- Individual model probabilities
- SHAP-based local explanations
- Important factors increasing or decreasing the predicted risk
- Human-readable explanations generated using Generative AI
- User authentication using Firebase
- Prediction history storage using Firestore
- Secure protected routes for authenticated users

> **Note:** This project is intended for research, educational, and screening-support purposes only. It is not a medical diagnostic system and should not replace professional medical advice.
# 🧠 Project Workflow

```text
User Input
    │
    ▼
React Frontend
    │
    ▼
FastAPI Backend
    │
    ├── Feature Processing
    │
    ▼
Ensemble Machine Learning Model
    │
    ├── Logistic Regression
    ├── Random Forest
    └── XGBoost
    │
    ▼
Cardiovascular Risk Prediction
    │
    ├── Predicted Probability
    ├── Risk Category
    └── Individual Model Predictions
    │
    ▼
SHAP Explainable AI
    │
    ├── Important Risk-Increasing Factors
    └── Important Risk-Decreasing Factors
    │
    ▼
Generative AI
    │
    ▼
Human-Readable Prediction Report
    │
    ▼
Firebase / Firestore
    │
    ▼
Prediction History
🤖 Machine Learning Approach

The project uses an ensemble of three machine learning models:

Logistic Regression
Random Forest
XGBoost

Each model generates a probability of cardiovascular risk.

The final ensemble prediction is calculated using weighted probabilities:

Final Probability =
(Logistic Regression Weight × Logistic Probability)
+
(Random Forest Weight × Random Forest Probability)
+
(XGBoost Weight × XGBoost Probability)

The final probability is then compared with predefined thresholds to determine the predicted risk category.

🔍 Explainable AI

The project uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

SHAP helps identify:

Which features increased the predicted risk
Which features decreased the predicted risk
The relative importance of different features
The influence of features on the XGBoost model prediction

The application provides local explanations for each individual assessment.

SHAP explains model behavior and feature influence. It does not establish medical causation.

✨ Generative AI

Generative AI is used to transform the technical machine learning and SHAP outputs into a clear and human-readable prediction report.

The generated report includes:

Risk summary
Predicted probability
Important contributing factors
Factors associated with lower predicted risk
Explanation of the prediction results
Appropriate interpretation notes

This makes the system easier for non-technical users to understand.

🛠️ Technologies Used
Frontend
React
TypeScript
Vite
CSS
React Router
Firebase Authentication
Lucide React
Backend
Python
FastAPI
Uvicorn
Pydantic
Machine Learning
Scikit-learn
XGBoost
Pandas
NumPy
Joblib
Explainable AI
SHAP
Generative AI
Google Gemini API
Database and Authentication
Firebase Authentication
Firebase Admin SDK
Cloud Firestore
Version Control
Git
GitHub
Git Large File Storage (Git LFS)
📂 Project Structure
IEEE-SMC-KGEC-Internship-2026/
│
├── heart-risk-xgenai/
│   │
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── heart_risk_model_bundle.joblib
│   │   │   │   └── model_environment.json
│   │   │   │
│   │   │   ├── schemas/
│   │   │   │   └── prediction.py
│   │   │   │
│   │   │   └── services/
│   │   │       ├── auth_service.py
│   │   │       ├── feature_service.py
│   │   │       ├── firebase_service.py
│   │   │       ├── gemini_service.py
│   │   │       ├── model_service.py
│   │   │       ├── prediction_database_service.py
│   │   │       └── shap_service.py
│   │   │
│   │   └── requirements.txt
│   │
│   └── frontend/
│       ├── public/
│       ├── src/
│       │   ├── components/
│       │   ├── context/
│       │   ├── lib/
│       │   ├── pages/
│       │   └── services/
│       │
│       ├── .env.example
│       ├── package.json
│       └── vite.config.ts
│
├── IEEE-SMC-SBC-KGEC-[2].docx
├── requirements.txt
└── README.md
👥 Team Members
Name	Role
Souren Mondal	AI/ML Developer & Full-Stack Developer
[Team Member 2]	[Role]
[Team Member 3]	[Role]

Update the team member names and roles according to your internship team.

🚀 Installation and Setup
Prerequisites

Make sure the following software is installed:

Python 3.10 or above
Node.js and npm
Git
Git LFS

Check the installations:

python --version
node --version
npm --version
git --version
git lfs version
1. Clone the Repository
git clone https://github.com/Souren44/IEEE-SMC-KGEC-Internship-2026.git

Move into the project:

cd IEEE-SMC-KGEC-Internship-2026
cd heart-risk-xgenai
2. Set Up Git LFS

The trained machine learning model is stored using Git LFS.

Run:

git lfs install
git lfs pull
3. Backend Setup

Move into the backend folder:

cd backend

Create a virtual environment:

python -m venv venv
Windows
venv\Scripts\activate
Linux/macOS
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
4. Configure Backend Environment Variables

Create a .env file inside the backend directory.

Example:

GEMINI_API_KEY=your_gemini_api_key
FIREBASE_PROJECT_ID=your_firebase_project_id

Do not upload the .env file to GitHub.

You also need your Firebase Admin SDK service account credentials configured according to the backend Firebase configuration.

Firebase service account credentials must remain private and should never be committed to GitHub.

5. Run the Backend

From the backend directory:

uvicorn app.main:app --reload

The FastAPI server should start at:

http://127.0.0.1:8000

API documentation is available at:

http://127.0.0.1:8000/docs
6. Frontend Setup

Open another terminal and move into:

cd heart-risk-xgenai/frontend

Install dependencies:

npm install

Create a .env file based on .env.example.

Example:

VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_storage_bucket
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id


VITE_API_URL=http://127.0.0.1:8000

Do not upload your actual .env file to GitHub.

7. Run the Frontend

From the frontend directory:

npm run dev

The application will typically run at:

http://localhost:5173
📖 Usage
Open the frontend application in your browser.
Register for a new account or log in.
Verify your email if required.
Navigate to the cardiovascular risk assessment page.
Enter the required health and lifestyle information.
Submit the assessment.
The backend processes the features.
The ensemble model calculates the predicted probability.
SHAP identifies the important factors influencing the XGBoost prediction.
Generative AI produces a human-readable explanation.
The prediction can be saved to the authenticated user's history.
Users can view previous prediction details.
🔐 Security and Privacy

The following files and folders are excluded from version control:

.env
firebase-service-account.json
firebase-adminsdk-*.json
serviceAccountKey.json
venv/
node_modules/

Sensitive credentials must never be committed to the repository.

⚠️ Disclaimer

This project is developed for:

Academic research
Educational purposes
Machine learning experimentation
Cardiovascular risk screening research

It is not a clinical diagnostic tool.

The predictions generated by this system should not be used as a substitute for professional medical diagnosis, treatment, or advice.

🔮 Future Improvements

Possible future improvements include:

Interactive SHAP visualizations
SHAP waterfall plots
SHAP force plots
PDF prediction reports
Downloadable health reports
Improved ensemble calibration
Additional cardiovascular datasets
More comprehensive clinical validation
Cloud deployment
Mobile application support
Personalized lifestyle recommendations
Advanced model monitoring
📄 Internship

This project was developed as part of the:

IEEE SMC Student Branch Chapter, KGEC Internship Program – 2026

👨‍💻 Author

Souren Mondal

B.Tech in Computer Science and Engineering
Specialization: Artificial Intelligence and Machine Learning

GitHub: Souren44
