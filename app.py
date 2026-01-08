import os, sys
import certifi
ca = certifi.where()

from src.exception.exception import NetWorkSecurityException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from src.utils.ml_utils.model.estimator import NetworkModel

from src.utils.main_utils.utils import load_object
from fastapi.responses import HTMLResponse

app = FastAPI()
origin = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetWorkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        # Load model components
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        # Make predictions
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        
        # Map predictions to readable labels
        df['prediction_label'] = df['predicted_column'].map({
            1.0: 'Phishing/Malicious',
            0.0: 'Legitimate',
            -1.0: 'Suspicious'
        })
        
        # Calculate statistics
        total_urls = len(df)
        phishing_count = (df['predicted_column'] == 1.0).sum()
        legitimate_count = (df['predicted_column'] == 0.0).sum()
        suspicious_count = (df['predicted_column'] == -1.0).sum()
        
        phishing_pct = (phishing_count / total_urls * 100) if total_urls > 0 else 0
        legitimate_pct = (legitimate_count / total_urls * 100) if total_urls > 0 else 0
        suspicious_pct = (suspicious_count / total_urls * 100) if total_urls > 0 else 0
        
        # Save output
        df.to_csv('prediction_output/output.csv', index=False)
        
        # Create summary statistics
        summary_stats = {
            'total': total_urls,
            'phishing': phishing_count,
            'legitimate': legitimate_count,
            'suspicious': suspicious_count,
            'phishing_pct': round(phishing_pct, 2),
            'legitimate_pct': round(legitimate_pct, 2),
            'suspicious_pct': round(suspicious_pct, 2)
        }
        
        # Prepare data for template
        table_html = df.to_html(classes='table table-striped', index=False)
        
        # Return proper HTML response with correct media_type
        return templates.TemplateResponse(
            "enhanced_table.html", 
            {
                "request": request,
                "table": table_html,
                "stats": summary_stats
            },
            media_type="text/html"  # This ensures proper HTML rendering
        )
        
    except Exception as e:
        raise NetWorkSecurityException(e, sys)

# Add this new route for the upload form
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        # Load model components
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        # Make predictions
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        
        # Map predictions to readable labels
        df['prediction_label'] = df['predicted_column'].map({
            1.0: 'Phishing/Malicious',
            0.0: 'Legitimate',
            -1.0: 'Suspicious'
        })
        
        # Calculate statistics
        total_urls = len(df)
        phishing_count = (df['predicted_column'] == 1.0).sum()
        legitimate_count = (df['predicted_column'] == 0.0).sum()
        suspicious_count = (df['predicted_column'] == -1.0).sum()
        
        phishing_pct = (phishing_count / total_urls * 100) if total_urls > 0 else 0
        legitimate_pct = (legitimate_count / total_urls * 100) if total_urls > 0 else 0
        suspicious_pct = (suspicious_count / total_urls * 100) if total_urls > 0 else 0
        
        # Save output
        df.to_csv('prediction_output/output.csv', index=False)
        
        # Create summary statistics
        summary_stats = {
            'total': total_urls,
            'phishing': phishing_count,
            'legitimate': legitimate_count,
            'suspicious': suspicious_count,
            'phishing_pct': round(phishing_pct, 2),
            'legitimate_pct': round(legitimate_pct, 2),
            'suspicious_pct': round(suspicious_pct, 2)
        }
        
        # Prepare data for template
        table_html = df.to_html(classes='table table-striped', index=False)
        
        # Return HTML response
        return templates.TemplateResponse(
            "enhanced_table.html", 
            {
                "request": request,
                "table": table_html,
                "stats": summary_stats
            }
        )
        
    except Exception as e:
        raise NetWorkSecurityException(e, sys)    
    
if __name__=='__main__':
    app_run(app, host='localhost', port=8000)    