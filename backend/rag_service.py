import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Simulated Western Digital Catalog
CATALOG_TEXT = """
1. Ultrastar DC HC550
   - Description: 18TB HDD for Data Centers/Hyperscale. High capacity, low TCO.
   - Ideal For: Cloud providers, Hyperscale data centers, Big Data analytics.
   - Key Features: HelioSeal technology, Energy-assisted magnetic recording (EAMR).

2. WD_BLACK SN850X
   - Description: NVMe SSD for Gaming/High-Performance. Blazing fast speeds.
   - Ideal For: Gamers, PC custom builders, detailed content creators.
   - Key Features: PCIe Gen4 technology, Game Mode 2.0, Up to 7300 MB/s.

3. SanDisk Professional G-DRIVE
   - Description: External storage for Creative Professionals. rugged, reliable.
   - Ideal For: Photographers, Videographers, Content Creators on the go.
   - Key Features: Thunderbolt 3, USB-C, Enterprise-class Ultrastar drive inside.
"""

# Simple Retrieval Logic (Keyword based for prototype to remove heavy dependencies like ChromaDB if not needed, 
# but valid RAG usually uses embeddings. Given user requirements for "RAG", I will implement a lightweight vector search or just a smart prompt selection 
# if embeddings are too heavy. Actually, let's use a simple heuristic "Retrieval" since the catalog is tiny (3 items).
# A full vector store for 3 items is overkill and prone to setup errors with environment keys. 
# I will implement a "Keyword/Rule Based Retriever" that mimics RAG behavior effectively for this scale.)

def retrieve_product(lead_profile: dict) -> str:
    """
    Retrieves relevant product based on Lead Profile rules.
    This simulates the 'Retrieval' step in RAG.
    """
    
    # 1. Extract Signals
    tags = str(lead_profile.get('Tags', '')).lower()
    source = str(lead_profile.get('Lead Source', '')).lower()
    specialization = str(lead_profile.get('Specialization', '')).lower() # If available
    
    # 2. Heuristic Matching (Simulating Vector Similarity)
    if "games" in tags or "hardware" in specialization:
        return "WD_BLACK SN850X (NVMe SSD for Gaming)"
    elif "data" in tags or "business" in tags or "logic" in source: # 'logic' -> generic business
        return "Ultrastar DC HC550 (18TB HDD for Data Centers)"
    elif "media" in tags or "creative" in tags or "design" in specialization:
        return "SanDisk Professional G-DRIVE (External Storage)"
    
    # Default fallback
    return "Ultrastar DC HC550 (Standard B2B Offering)"

def generate_email_content(lead_profile: dict, product_name: str, product_details: str):
    """
    Generates an email using an LLM.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set. Please set it to generate emails."

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    
    template = """
    You are a professional B2B Sales Representative for Western Digital.
    
    Goal: Write a personalized cold email to a prospective lead.
    
    Lead Context:
    - Time on Website: {time_on_site} seconds (High engagement implies interest).
    - Source: {source}
    - Tags: {tags}
    
    Recommended Product: {product_name}
    Product Details: {product_details}
    
    Instructions:
    - Acknowledge their interest (based on time on site).
    - Pitch the recommended product as a solution to their potential needs (inferred from tags/source).
    - Keep it professional, concise, and persuasive.
    - End with a call to action.
    
    Email Draft:
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "time_on_site": lead_profile.get('Total Time Spent on Website', 0),
        "source": lead_profile.get('Lead Source', 'Website'),
        "tags": lead_profile.get('Tags', 'General Interest'),
        "product_name": product_name,
        "product_details": product_details
    })
    
    return response

def get_product_details(product_name):
    # Retrieve full details text from the catalog string
    if "Ultrastar" in product_name:
        return "18TB HDD for Data Centers/Hyperscale. High capacity, low TCO. Ideal for Cloud providers."
    elif "WD_BLACK" in product_name:
        return "NVMe SSD for Gaming/High-Performance. Blazing fast speeds. Ideal for Gamers and PC builders."
    elif "SanDisk" in product_name:
        return "External storage for Creative Professionals. Rugged, reliable. enterprise-class Ultrastar drive inside."
    return "High quality storage solution from Western Digital."

def run_rag_pipeline(lead_profile: dict):
    product = retrieve_product(lead_profile)
    details = get_product_details(product)
    email = generate_email_content(lead_profile, product, details)
    return {
        "recommended_product": product,
        "email_draft": email
    }
