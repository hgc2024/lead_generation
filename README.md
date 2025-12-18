# Hybrid AI Sales Agent

## Project Overview
This project is a **Hybrid AI Sales Agent** prototype designed to demonstrate how Artificial Intelligence can augment B2B sales workflows. It combines traditional **predictive analytics** (Machine Learning) with **Generative AI** (RAG) to identify high-value leads and draft personalized outreach emails automatically.

The system simulates a Western Digital sales environment, recommending products like **Ultrastar HDDs**, **WD_BLACK SSDs**, and **SanDisk Professional** drives based on lead behavior.

## Key Features
1.  **Predictive Lead Scoring (Phase A)**:
    *   Uses a **Random Forest Classifier** to analyze lead behavior (Time on Website, Source, Tags).
    *   Outputs a conversion probability score for each lead.
    *   Identifies key drivers of conversion (Feature Importance).
2.  **RAG-Based Email Generation (Phase B)**:
    *   **retrieval**: Matches lead profiles to specific products in the Western Digital catalog.
    *   **Generation**: Uses **Google Gemini Pro** (via LangChain) to draft highly personalized sales emails referencing the lead's specific interests and the matched product's unique value proposition.
3.  **ROI Analysis (Phase C)**:
    *   Calculates the estimated cost savings of using AI for email drafting compared to manual SDR efforts.

## Technology Stack
*   **Backend**: Python, FastAPI
*   **Machine Learning**: Scikit-Learn, Pandas
*   **Generative AI**: LangChain, Google Gemini API
*   **Frontend**: React, Vite, Recharts, Lucide React
*   **Styling**: Vanilla CSS (Dark Mode / Glassmorphism)

## Project Structure
```
lead_generation/
├── backend/
│   ├── main.py          # FastAPI Entry Point
│   ├── ml_service.py    # ML Model Training & Inference
│   ├── rag_service.py   # RAG Logic (Retrieval + Generation)
│   ├── models.py        # Pydantic Data Models
│   └── requirements.txt # Python Dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx      # Main Dashboard Logic
│   │   └── ...
│   └── ...
├── data/
│   └── archive/         # Dataset Location
├── run_project.bat      # One-click startup script
└── setup_venv.bat       # Environment setup script
```

## Getting Started

### Prerequisites
*   **Python 3.9+**
*   **Node.js 16+**
*   **Google Gemini API Key**: Required for email generation.

### Installation & Running
1.  **Clone/Open the Repository**.
2.  **Set API Key**:
    *   Set the `GEMINI_API_KEY` environment variable in your terminal.
    *   *(Windows Powershell)*: `$env:GEMINI_API_KEY="your_api_key_here"`
    *   *(Command Prompt)*: `set GEMINI_API_KEY=your_api_key_here`
3.  **Run the Application**:
    *   Double-click `run_project.bat` (or run it from the terminal).
    *   This script will automatically:
        *   Create a Python virtual environment (`venv`).
        *   Install all Backend dependencies.
        *   Install all Frontend node modules.
        *   Launch the **Backend (Port 8000)** and **Frontend (Port 5173)**.

### Usage
1.  Open your browser to `http://localhost:5173`.
2.  **Dashboard**: View the list of leads sorted by "Conversion Probability".
3.  **Analytics**: Check the "Top Drivers of Conversion" chart to see what influences sales.
4.  **Generate Email**: Click **"Draft Email"** on any lead to see the RAG agent in action.

## License
MIT