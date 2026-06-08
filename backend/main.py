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

# --- Constraint extraction via LLM ---
constraints_parser = JsonOutputParser(pydantic_object=None)

# We'll create a PromptTemplate that asks the LLM to return a JSON object matching
# the `RecommendationConstraints` fields. We include the list of allowed category
# tokens so the model must pick one of them (or null).
constraints_template = """
You are an assistant that extracts structured search constraints from a user's free-text query.
Allowed categories: {allowed_categories}

Return ONLY a JSON object with these fields (use null when not applicable):
- category: one of the allowed categories (exact match) or null
- max_price: number or null
- min_price: number or null

Be strict: do not output any other keys or explanatory text. Output must be valid JSON.

User Query: {query}
{format_instructions}
"""

from langchain_core.output_parsers import JsonOutputParser as _JsonParser
from langchain_core.prompts import PromptTemplate as _PromptTemplate

# Create a minimal JsonOutputParser by providing the format instructions text
# from our Pydantic model shape so the LLM knows the expected schema.
constraints_parser = JsonOutputParser(pydantic_object=None)
constraints_prompt = PromptTemplate(
    template=constraints_template,
    input_variables=["query", "allowed_categories"],
    partial_variables={"format_instructions": "Please return JSON with keys: category, max_price, min_price."}
)

# Chain for constraints: prompt -> llm -> parser
# We'll invoke the LLM directly and parse the JSON response into the Pydantic model.


# Parsed constraints derived from the user's natural language query.
class RecommendationConstraints(BaseModel):
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None

    class Config:
        extra = 'forbid'


def parse_constraints(q: str) -> RecommendationConstraints:
    ql = q.lower()
    c: dict = {}
    # category detection
    if any(tok in ql for tok in ("smartphone", "smartphones", "phone", "phones", "mobile", "mobiles")):
        c['category'] = 'phone'
    elif any(tok in ql for tok in ("audio", "headphone", "headphones", "speaker", "speakers")):
        c['category'] = 'audio'
    elif any(tok in ql for tok in ("computer", "laptop", "notebook")):
        c['category'] = 'computers'

    # budget / price hints
    if any(term in ql for term in ("budget", "cheap", "affordable", "inexpensive", "low cost", "low-cost")):
        c['max_price'] = 500.0
    if any(term in ql for term in ("high-end", "premium", "expensive", "high end")):
        c['min_price'] = 500.0

    # explicit numeric price constraints like "under $300" or "below 400"
    import re
    m_under = re.search(r"(?:under|below|less than|<)\s*\$?(\d{2,5})", ql)
    if m_under:
        try:
            c['max_price'] = float(m_under.group(1))
        except:
            pass
    m_over = re.search(r"(?:over|above|more than|>)\s*\$?(\d{2,5})", ql)
    if m_over:
        try:
            c['min_price'] = float(m_over.group(1))
        except:
            pass

    return RecommendationConstraints(**c)


def apply_constraints(items: List[dict], constraints: RecommendationConstraints) -> List[dict]:
    if not constraints:
        return items
    out = []
    for p in items:
        text = (p['name'] + ' ' + p['description'] + ' ' + p['category']).lower()
        import re
        words = set(re.findall(r"\w+", text))
        # category matching
        if constraints.category:
            cat = constraints.category
            if cat == 'phone':
                phone_tokens = {"phone", "phones", "smartphone", "smartphones", "mobile", "mobiles", "cellphone", "cellphones"}
                if not (words & phone_tokens):
                    continue
            else:
                if cat not in text:
                    continue
        # price matching
        if constraints.max_price is not None and p.get('price') is not None:
            if float(p['price']) > float(constraints.max_price):
                continue
        if constraints.min_price is not None and p.get('price') is not None:
            if float(p['price']) < float(constraints.min_price):
                continue
        out.append(p)
    return out

@app.post("/api/recommend", response_model=List[Product])
async def recommend_products(request: RecommendationRequest):
    # First try to extract structured constraints using the LLM (preferred).
    # If the LLM extraction fails, fall back to the rule-based parser.
    allowed_categories = ["phone", "audio", "accessories", "computers", "gaming", "electronics"]
    try:
        constraints_parser = JsonOutputParser(pydantic_object=RecommendationConstraints)
        chain_constraints = constraints_prompt | llm | constraints_parser
        constraints_response = chain_constraints.invoke({
            "query": request.query,
            "allowed_categories": json.dumps(allowed_categories)
        })
        # Normalize into RecommendationConstraints
        if hasattr(constraints_response, 'dict'):
            constraints = RecommendationConstraints(**constraints_response.dict())
        elif isinstance(constraints_response, dict):
            constraints = RecommendationConstraints(**constraints_response)
        else:
            # try parsing raw JSON text
            try:
                parsed = json.loads(str(constraints_response))
                constraints = RecommendationConstraints(**parsed)
            except Exception:
                constraints = parse_constraints(request.query)
    except Exception:
        constraints = parse_constraints(request.query)
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

        # If the LLM returned no recommendations, fall back to a simple heuristic
        # to provide useful results during development or when the LLM is unavailable.
        if not matched_products:
            q = request.query.lower()
            # budget-oriented fallback
            budget_terms = ["budget", "cheap", "affordable", "inexpensive", "low cost", "low-cost"]
            fallback = []
            if any(term in q for term in budget_terms):
                # prefer phones and lower-priced items
                fallback = [p for p in products if ("phone" in (p["name"] + p["description"]).lower()) or p["price"] <= 500]
            else:
                # token-match fallback: match words in name, description, or category
                tokens = [t for t in q.split() if len(t) > 2]
                for p in products:
                    text = (p["name"] + " " + p["description"] + " " + p["category"]).lower()
                    if any(t in text for t in tokens):
                        fallback.append(p)

            # dedupe and limit results
            seen = set()
            deduped = []
            for p in fallback:
                if p["id"] not in seen:
                    deduped.append(p)
                    seen.add(p["id"])
                if len(deduped) >= 6:
                    break

            matched_products = deduped

        # Enforce parsed constraints strictly: return only items that match user's explicit intent
        matched_products = apply_constraints(matched_products, constraints)

        return matched_products

    except Exception as e:
        print(f"Error during recommendation: {e}")
        # If the LLM call fails, fall back to the same heuristic used above
        q = request.query.lower() if hasattr(request, 'query') else ''
        budget_terms = ["budget", "cheap", "affordable", "inexpensive", "low cost", "low-cost"]
        fallback = []
        if any(term in q for term in budget_terms):
            fallback = [p for p in products if ("phone" in (p["name"] + p["description"]).lower()) or p["price"] <= 500]
        else:
            tokens = [t for t in q.split() if len(t) > 2]
            for p in products:
                text = (p["name"] + " " + p["description"] + " " + p["category"]).lower()
                if any(t in text for t in tokens):
                    fallback.append(p)

        # dedupe and limit
        seen = set()
        deduped = []
        for p in fallback:
            if p["id"] not in seen:
                deduped.append(p)
                seen.add(p["id"])
            if len(deduped) >= 6:
                break

        # Apply constraints here as well before returning
        return apply_constraints(deduped, constraints)

@app.get("/api/products", response_model=List[Product])
async def get_all_products():
    return products

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
