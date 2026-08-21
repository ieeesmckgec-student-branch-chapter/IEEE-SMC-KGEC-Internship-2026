# ProsperityScore

> **An Alternative Credit Scoring Platform For Lending Informal Workforce**

ProsperityScore is a full-stack prototype designed to analyze bank statements and generate an explainable financial assessment. The platform processes transaction data from **CSV, XLS, and XLSX** files, extracts relevant financial features, computes component-wise scores, generates a final **Prosperity Score**, classifies financial risk, and evaluates loan eligibility.

The project provides a guided user workflow:

**Identity → Bank Statement Upload → Profile Review → Financial Assessment Results**

---

## 👥 Team

- **Krishnendu Ghosh** BTech AIML Final Year Supreme Knowledge Foundation , Chandanagar, Hooghly
- **Animikh Chowdhury** B.Sc Computer Science 2nd Year St. Xavier Autonomus Kolkata

**Mentor:** Sajal Bhadra ,Consultant, TATA Consultancy Service Limited

---

## 🚀 Live Demo

**Frontend:**  
https://prosperityscore.vercel.app/

**Backend API:**  
https://purchase-capability-ieee-kgec.onrender.com

---

## ✨ Features

- Upload bank statements in **CSV, XLS, and XLSX** formats
- Automated transaction preprocessing and analysis
- Financial feature extraction
- Component-wise credit assessment
- Final **Prosperity Score** generation
- Risk bucket classification
- Loan eligibility evaluation
- Statement-based financial summary generation
- Automated profile field autofill
- Guided user workflow and profile review
- Explainable financial assessment results
- Cross-origin API integration with CORS and cookie support

---

## 📊 Scoring Parameters

The system analyzes multiple financial dimensions, including:

- **Income**
- **Expenses**
- **Cashflow**
- **Repayment Behaviour**
- **Financial Behaviour**

These indicators are processed to generate component scores and an aggregated **Prosperity Score**, along with a risk classification and eligibility assessment.

---

## 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Bank Statement    │
                │ CSV / XLS / XLSX    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Transaction Parsing │
                │ & Preprocessing     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Extraction  │
                │ Income • Expense    │
                │ Cashflow • Repayment│
                │ Behaviour           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Scoring Engine    │
                │ Component Scores    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prosperity Score    │
                │ Risk • Eligibility  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ React Frontend      │
                │ Review & Results    │
                └─────────────────────┘
```

---

## 🛠️ Technologies Used

### Frontend

- React
- TypeScript
- Vite
- TanStack Router
- Tailwind CSS

### Backend

- Python
- FastAPI
- pandas
- NumPy

### Development Tools

- Git
- VS Code
- npm / Bun

### Deployment

- **Frontend:** Vercel
- **Backend:** Render

---

# ⚙️ Installation and Setup

## Prerequisites

Make sure you have the following installed:

- Node.js 20+
- npm or Bun
- Python 3.10+
- Git

---

## A. Quick Setup — Frontend with Hosted Backend

Clone the repository:

```bash
git clone
```

Navigate to the frontend directory:

```bash
cd Prosperity-Frontend
```

Create a `.env` file:

```env
VITE_API_URL=https://purchase-capability-ieee-kgec.onrender.com
```

Install dependencies:

```bash
bun install
```

Or using npm:

```bash
npm install
```

Start the development server:

```bash
bun dev
```

Or:

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

---

# 🖥️ Full Local Setup

## 1. Backend

Navigate to the backend directory:

```bash
cd Purchase-Capability-IEEE-KGEC
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn api:app --reload --port 8000
```

Verify the backend:

```bash
curl http://localhost:8000/health
```

The backend should be available at:

```text
http://localhost:8000
```

---

## 2. Frontend

Navigate to the frontend directory:

```bash
cd Prosperity-Frontend
```

Update `.env`:

```env
VITE_API_URL=http://localhost:8000
```

Install dependencies:

```bash
bun install
```

Start the development server:

```bash
bun dev
```

Or:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

---

# 📁 Supported Bank Statement Formats

The system currently supports:

- `.csv`
- `.xls`
- `.xlsx`

---

# 🔄 Application Workflow

1. User enters the required profile information.
2. User uploads a bank statement.
3. The frontend sends the file to the backend API.
4. The backend parses and preprocesses transaction data.
5. Financial features are extracted from the transactions.
6. Component-wise scores are calculated.
7. The final Prosperity Score is generated.
8. The system classifies the user's risk level.
9. Loan eligibility is evaluated.
10. A compact statement summary is generated.
11. Relevant financial fields are automatically populated in the profile review flow.
12. The user reviews the information and views the final assessment.

---

# 🔌 API Usage

## Score Bank Statement

**Endpoint:**

```http
POST /score
```

Example request:

```bash
curl -X POST \
"https://purchase-capability-ieee-kgec.onrender.com/score" \
-F "file=@/path/to/statement.csv"
```

The API processes the uploaded statement and returns financial scoring results, including:

- Component scores
- Final Prosperity Score
- Risk bucket
- Loan eligibility
- Statement summary for profile autofill

---

## Get Statement Summary

**Endpoint:**

```http
GET /api/statement/summary?id=<uuid>
```

Example:

```bash
curl "https://purchase-capability-ieee-kgec.onrender.com/api/statement/summary?id=<uuid>"
```

The summary may contain inferred financial information such as:

- Monthly income
- Average bank balance
- Rent
- EMI obligations
- Other loans
- Employment type
- Utility payment indicators

---

# 🔐 Cookie and CORS Support

The backend may generate an HttpOnly cookie named:

```text
statement_id
```

When frontend functionality depends on cookies, requests should include credentials:

```typescript
credentials: "include";
```

CORS configuration should allow the required frontend origins and support credentialed requests.

---

# 🧪 Testing

The project was tested across multiple stages, including:

- Bank statement parsing
- Transaction preprocessing
- Feature extraction
- Component score calculation
- Final score generation
- Risk classification
- Loan eligibility evaluation
- API integration
- Frontend-backend communication
- Statement autofill functionality
- CORS and cookie behavior

---

# 🔮 Future Improvements

- Add confidence scores for inferred financial attributes
- Improve employment and transaction classification
- Replace hard inference fallbacks with nullable predictions where appropriate
- Integrate persistent database storage
- Add audit logs for financial assessments
- Develop an administrative dashboard
- Train ML models using labeled financial datasets
- Improve explainability and visualization of component scores

---

# 📌 Project Outcome

ProsperityScore demonstrates an end-to-end pipeline for **alternative credit assessment using bank transaction data**. The system combines financial data processing, feature engineering, scoring logic, explainability, API development, and an interactive frontend to transform a raw bank statement into a structured financial assessment.

---

## 👨‍💻 Team

| Name                  | Role                |
| --------------------- | ------------------- | ---------------------------------------------------------------- |
| **Krishnendu Ghosh**  | Project Team Member | BTech AIML Final Year Supreme Knowledge Foundation , Chandanagar |
| **Animikh Chowdhury** | Project Team Member | B.Sc Computer Science 2nd Year St Xavier Autonomus Kolkata       |

2
**Mentor:** Sajal Bhadra

---

⭐ If you find this project interesting, consider giving the repository a star!
