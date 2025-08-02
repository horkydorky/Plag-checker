# main.py

import os
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mistralai.client import MistralClient
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This line loads the MISTRAL_API_KEY from your .env file
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# --- Error Handling & Initialization ---
if not api_key:
    raise ValueError("CRITICAL ERROR: The MISTRAL_API_KEY environment variable is not set. Please create a .env file.")

app = FastAPI(title="Professional Plagiarism Checker API")
try:
    client = MistralClient(api_key=api_key)
    print("Successfully connected to Mistral AI.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize MistralClient: {e}")

# --- Pydantic Models for API Data Structure ---
class PlagiarismCheckRequest(BaseModel):
    source_documents: list[str]
    submitted_text: str
    distance_threshold: float = 0.2

class MatchDetail(BaseModel):
    submitted: str
    source: str
    distance: float

class AnalysisResult(BaseModel):
    match_details: MatchDetail
    expert_verdict: str

class PlagiarismCheckResponse(BaseModel):
    status: str
    results: list[AnalysisResult]

# --- The Main API Endpoint ---
@app.post("/check-plagiarism/", response_model=PlagiarismCheckResponse)
def check_plagiarism_endpoint(request: PlagiarismCheckRequest):
    """
    Receives source documents and a submitted text, performs a two-stage
    plagiarism check, and returns a structured analysis.
    """
    try:
        # --- PART 1: BROAD SEARCH (Vector Search) ---
        source_embeddings = client.embeddings(
            model="mistral-embed",
            input=request.source_documents
        ).data
        source_vectors = np.array([data.embedding for data in source_embeddings])

        index = faiss.IndexFlatL2(source_vectors.shape[1])
        index.add(source_vectors)

        submitted_sentences = request.submitted_text.strip().split('\n')
        potential_matches = []

        for sentence in submitted_sentences:
            if not sentence: continue
            query_embedding = client.embeddings(model="mistral-embed", input=[sentence]).data[0].embedding
            query_vector = np.array([query_embedding])

            distances, indices = index.search(query_vector, 1)
            match_distance = distances[0][0]

            if match_distance < request.distance_threshold:
                potential_matches.append({
                    "submitted": sentence,
                    "source": request.source_documents[indices[0][0]],
                    "distance": float(match_distance)
                })

        # --- PART 2: DETAILED ANALYSIS (LLM Judgment) ---
        analysis_results = []
        if potential_matches:
            for match in potential_matches:
                prompt = f"""You are a plagiarism detection expert. Analyze the following texts and provide a verdict (e.g., Clear Paraphrase) and a brief reason.\n\nSource: "{match['source']}"\n\nSubmitted: "{match['submitted']}" """
                
                chat_response = client.chat(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                verdict = chat_response.choices[0].message.content
                analysis_results.append({
                    "match_details": match,
                    "expert_verdict": verdict
                })

        return {"status": "success", "results": analysis_results}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")