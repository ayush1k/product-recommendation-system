from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Product Recommender")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mock Database
products = [
    {"id": 1, "name": "iPhone 13 Mini", "price": 599, "category": "Electronics", "description": "Compact and powerful smartphone with a great camera."},
    {"id": 2, "name": "Samsung Galaxy A54", "price": 449, "category": "Electronics", "description": "Budget-friendly 5G phone with a vibrant display."},
    {"id": 3, "name": "Sony WH-1000XM5", "price": 399, "category": "Audio", "description": "Industry-leading noise-canceling headphones."},
    {"id": 4, "name": "Kindle Paperwhite", "price": 139, "category": "Electronics", "description": "The best e-reader for book lovers."},
    {"id": 5, "name": "Logitech MX Master 3S", "price": 99, "category": "Accessories", "description": "Ergonomic wireless mouse for productivity."},
    {"id": 6, "name": "Dell XPS 13", "price": 999, "category": "Computers", "description": "Sleek and powerful ultrabook for professionals."},
    {"id": 7, "name": "JBL Flip 6", "price": 129, "category": "Audio", "description": "Portable waterproof Bluetooth speaker."},
    {"id": 8, "name": "Anker PowerCore 26800", "price": 65, "category": "Accessories", "description": "High-capacity portable charger."},
    {"id": 9, "name": "Nintendo Switch OLED", "price": 349, "category": "Gaming", "description": "Versatile gaming console with a vibrant OLED screen."},
    {"id": 10, "name": "Blue Yeti USB Microphone", "price": 129, "category": "Audio", "description": "Professional-grade microphone for streaming and recording."}
]

# 2. Pydantic Models for API
class RecommendationRequest(BaseModel):
    query: str

class Product(BaseModel):
    id: int
    name: str
    price: float
    category: str
    description: str

# 3. LangChain Output Parser Model
class RecommendationOutput(BaseModel):
    recommended_ids: List[int] = Field(description="List of product IDs recommended for the user")

# 4. Set up LangChain Pipeline
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    max_new_tokens=128,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

parser = JsonOutputParser(pydantic_object=RecommendationOutput)

template = """
You are a product recommendation assistant. Given the following list of products and a user query, recommend the most relevant product IDs.
Return ONLY a JSON object with a key "recommended_ids" containing a list of integers.

Products:
{products_list}

User Query: {query}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["products_list", "query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

@app.post("/api/recommend", response_model=List[Product])
async def recommend_products(request: RecommendationRequest):
    try:
        # Stringify products for the prompt
        products_str = json.dumps(products, indent=2)
        
        # Execute chain
        response = chain.invoke({
            "products_list": products_str,
            "query": request.query
        })
        
        # Handle cases where the response might not be in the exact format
        recommended_ids = response.get("recommended_ids", [])
        
        # Filter products from database
        matched_products = [p for p in products if p["id"] in recommended_ids]
        
        return matched_products

    except Exception as e:
        print(f"Error during recommendation: {e}")
        # Return empty list as fallback
        return []

@app.get("/api/products", response_model=List[Product])
async def get_all_products():
    return products

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
