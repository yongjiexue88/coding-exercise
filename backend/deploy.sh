#!/bin/bash

# Configuration
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="rag-backend"
REGION="us-central1"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project selected. Run 'gcloud config set project [PROJECT_ID]' first."
    exit 1
fi

echo "🚀 Deploying $SERVICE_NAME to project $PROJECT_ID in $REGION..."

# Check for .env and source it if available
if [ -f .env ]; then
    echo "📄 Found .env file, sourcing variables..."
    export $(grep -v '^#' .env | xargs)
fi

# Validate API Key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY not found in environment."
    echo "   The service will deploy, but 'auto-ingest' will FAIL on startup without it."
    read -p "   Do you want to proceed anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Build and Submit Container
echo "📦 Building container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME .

# 2. Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 1 \
    --concurrency 80 \
    --set-env-vars "GEMINI_MODEL=gemini-2.0-flash,CHROMA_PERSIST_DIR=/tmp/chroma_db,GEMINI_API_KEY=$GEMINI_API_KEY"

echo "Please set your GEMINI_API_KEY manually if not set, or use:"
echo "gcloud run services update $SERVICE_NAME --set-env-vars GEMINI_API_KEY=[YOUR_KEY]"
