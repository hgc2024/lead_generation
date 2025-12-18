from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import TrainResponse, EmailGenerationRequest, EmailGenerationResponse
import backend.ml_service as ml_service
import backend.rag_service as rag_service

app = FastAPI(title="Hybrid AI Sales Agent API")

# Setup CORS for React Frontend (Port 5173 usually)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*" # For development ease
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Western Digital AI Sales Agent Backend is Running"}

@app.post("/api/train", response_model=TrainResponse)
def train_model_endpoint():
    try:
        results = ml_service.train_model()
        return TrainResponse(
            accuracy=results['accuracy'],
            feature_importance=results['feature_importance'],
            message="Model trained successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/leads")
def get_leads_endpoint():
    try:
        leads = ml_service.get_leads_data()
        return leads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-email", response_model=EmailGenerationResponse)
def generate_email_endpoint(request: EmailGenerationRequest):
    try:
        # Convert Pydantic model to dict
        lead_profile = request.lead_profile.dict()
        
        # Run RAG
        result = rag_service.run_rag_pipeline(lead_profile)
        
        return EmailGenerationResponse(
            email_content=result['email_draft'],
            product_recommended=result['recommended_product']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roi")
def calculate_roi_endpoint():
    # ROI Logic as per requirements
    # Input tokens = 500, Output tokens = 200. Price = $0.002/1k tokens.
    
    input_tokens = 500
    output_tokens = 200
    price_per_1k = 0.002
    
    cost_per_email = ((input_tokens + output_tokens) / 1000) * price_per_1k
    
    # Assuming human SDR takes 15 mins ($25/hr rate -> $6.25 per email)
    human_cost = 6.25
    savings_per_email = human_cost - cost_per_email
    
    return {
        "cost_per_email_ai": f"${cost_per_email:.5f}",
        "human_cost_baseline": f"${human_cost:.2f}",
        "savings_per_thousand_emails": f"${savings_per_email * 1000:.2f}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
