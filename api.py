from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging

# Import the ProductMatcher class (assuming it's in the same file or imported)
from hybridMatching import ProductMatcher

app = FastAPI(title="Product Matching API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the matcher
matcher = ProductMatcher()


# Request Models
class AmazonProduct(BaseModel):
    """Amazon product data model"""
    title: str
    category: Optional[str] = None
    # Allow any additional fields
    class Config:
        extra = "allow"


class GoogleSEOProduct(BaseModel):
    """Google SEO product data model"""
    title: str
    # Allow any additional fields
    class Config:
        extra = "allow"


class MatchRequest(BaseModel):
    """Request model for product matching"""
    amazon: AmazonProduct
    google_seo: List[GoogleSEOProduct] = Field(..., min_items=1)
    
    @validator('amazon')
    def validate_amazon_title(cls, v):
        if not v.title or v.title.strip() == "":
            raise ValueError("Amazon product title cannot be empty")
        return v
    
    @validator('google_seo')
    def validate_google_seo_titles(cls, v):
        if not v:
            raise ValueError("Google SEO products list cannot be empty")
        for idx, product in enumerate(v):
            if not product.title or product.title.strip() == "":
                raise ValueError(f"Google SEO product at index {idx} has empty title")
        return v


# Response Models
class MatchScore(BaseModel):
    """Individual score breakdown"""
    brand: float
    core_product: float
    size: float
    quantity: float
    color: float


class MatchResult(BaseModel):
    """Match result for a single product"""
    serp_title: str
    status: str  # "MATCH" or "REVIEW"
    confidence_score: float
    scores: MatchScore


class MatchResponse(BaseModel):
    """Response model for product matching"""
    amazon_product: str
    category: Optional[str] = None
    category_warning: Optional[str] = None
    best_matches: List[MatchResult] = []
    review_matches: List[MatchResult] = []
    total_serp_products: int
    total_matches_found: int
    total_reviews_found: int


@app.post("/api/v1/match-products", response_model=MatchResponse)
async def match_products(request: MatchRequest):
    """
    Match Amazon product against Google SEO products.
    
    Args:
        request: MatchRequest containing Amazon product and list of Google SEO products
        
    Returns:
        MatchResponse with best matches and products needing review
    """
    try:
        # Extract data
        amazon_title = request.amazon.title.strip()
        category = request.amazon.category.strip() if request.amazon.category else None
        
        # Check for category and add warning if missing
        category_warning = None
        if not category or category == "":
            category_warning = "Warning: Category not provided. Matching will proceed without category-specific rules."
            category = "General"  # Default category
            logger.warning(f"Category missing for product: {amazon_title}")
        
        # Extract Google SEO titles
        serp_titles = [product.title.strip() for product in request.google_seo]
        
        logger.info(f"Matching Amazon product: {amazon_title} against {len(serp_titles)} SERP products")
        
        # Perform matching
        match_results = matcher.match_against_multiple(
            amazon_title=amazon_title,
            serp_titles=serp_titles,
            category=category
        )
        
        # Separate results into best matches and review matches
        best_matches = []
        review_matches = []
        
        for serp_title, result in match_results:
            # Extract scores from match details
            scores = result.match_details.get('scores', {})
            
            match_data = MatchResult(
                serp_title=serp_title,
                status="MATCH" if result.is_match else "REVIEW" if result.needs_review else "NO_MATCH",
                confidence_score=round(result.confidence_score, 2),
                scores=MatchScore(
                    brand=round(scores.get('brand', 0.0), 2),
                    core_product=round(scores.get('core_product', 0.0), 2),
                    size=round(scores.get('size', 0.0), 2),
                    quantity=round(scores.get('quantity', 0.0), 2),
                    color=round(scores.get('color', 0.0), 2)
                )
            )
            
            # Categorize based on status
            if result.is_match:
                best_matches.append(match_data)
            elif result.needs_review:
                review_matches.append(match_data)
            # Ignore NO_MATCH results (below 80%)
        
        # Prepare response
        response = MatchResponse(
            amazon_product=amazon_title,
            category=category if category != "General" else None,
            category_warning=category_warning,
            best_matches=best_matches,
            review_matches=review_matches,
            total_serp_products=len(serp_titles),
            total_matches_found=len(best_matches),
            total_reviews_found=len(review_matches)
        )
        
        logger.info(f"Matching complete: {len(best_matches)} matches, {len(review_matches)} reviews")
        
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Unexpected error during matching: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during product matching: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Product Matching API"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Product Matching API",
        "version": "1.0.0",
        "endpoints": {
            "match": "/api/v1/match-products",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }


# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    """
    Example request:
    
    POST http://localhost:8000/api/v1/match-products
    
    {
        "amazon": {
            "title": "CeraVe Moisturizing Cream 16 oz Daily Face Moisturizer",
            "category": "Beauty",
            "price": 15.99,
            "asin": "B00TTD9BRC"
        },
        "google_seo": [
            {
                "title": "CeraVe Moisturizing Cream 16 oz",
                "price": 14.99,
                "url": "https://example.com/product1"
            },
            {
                "title": "CeraVe Moisturizing Cream 8 oz",
                "price": 9.99,
                "url": "https://example.com/product2"
            },
            {
                "title": "CeraVe Facial Moisturizing Lotion 16 oz",
                "price": 13.99,
                "url": "https://example.com/product3"
            }
        ]
    }
    
    Example response:
    
    {
        "amazon_product": "CeraVe Moisturizing Cream 16 oz Daily Face Moisturizer",
        "category": "Beauty",
        "category_warning": null,
        "best_matches": [
            {
                "serp_title": "CeraVe Moisturizing Cream 16 oz",
                "status": "MATCH",
                "confidence_score": 95.50,
                "scores": {
                    "brand": 1.0,
                    "core_product": 0.92,
                    "size": 1.0,
                    "quantity": 1.0,
                    "color": 1.0
                }
            }
        ],
        "review_matches": [
            {
                "serp_title": "CeraVe Facial Moisturizing Lotion 16 oz",
                "status": "REVIEW",
                "confidence_score": 85.30,
                "scores": {
                    "brand": 1.0,
                    "core_product": 0.75,
                    "size": 1.0,
                    "quantity": 1.0,
                    "color": 1.0
                }
            }
        ],
        "total_serp_products": 3,
        "total_matches_found": 1,
        "total_reviews_found": 1
    }
    """