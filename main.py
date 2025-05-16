from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from recommendation_model import AnimeRecommendationModel
import logging
from fastapi.middleware.cors import CORSMiddleware
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Anime Recommendation API", description="Public API to recommend similar anime based on genre and rating")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add your production frontend URL when deployed, e.g., "https://your-frontend.onrender.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Specific origins for your client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the model
model = None

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = AnimeRecommendationModel()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
        raise

# Define request models for input validation
class UserRecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

class GenreRecommendationRequest(BaseModel):
    genres: List[str]
    type_anime: str = "all"
    top_n: int = 5

@app.get("/image")
async def get_image(query: str):
    apiKey = "YOUR_GOOGLE_API_KEY"
    cseId = "YOUR_CSE_ID"
    response = await fetch(
        f"https://www.googleapis.com/customsearch/v1?key={apiKey}&cx={cseId}&q={query}&searchType=image&num=1"
    )
    return await response.json()
# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Anime Recommendation API"}

# Endpoint for recommending similar anime
@app.get("/recommend_similar_anime/")
async def recommend_similar_anime(
    anime_name: str = Query(..., description="Name of the anime to find recommendations for"),
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
    rating_threshold: float = Query(6.0, ge=0.0, le=10.0, description="Minimum rating for recommended anime")
):
    try:
        recommendations = model.recommend_similar_anime(anime_name, top_n, rating_threshold)
        if recommendations.empty:
            return {"message": "No recommendations found for the given criteria"}
        result = recommendations.to_dict(orient="records")
        return {"recommendations": result}
    except Exception as e:
        logger.error(f"Error in recommend_similar_anime: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for user-based recommendations
@app.post("/recommend/user")
async def recommend_by_user(request: UserRecommendationRequest):
    try:
        recommendations = model.get_recommendation(request.user_id, request.top_n)
        return recommendations.to_dict(orient="records")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User ID {request.user_id} not found")
    except Exception as e:
        logger.error(f"Error in recommend_by_user: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for genre-based recommendations
@app.post("/recommend/genre")
async def recommend_by_genre(request: GenreRecommendationRequest):
    try:
        popular_df, relevant_df = model.recommendation_by_genre(
            request.genres, request.type_anime, request.top_n
        )
        return {
            "popular": popular_df.to_dict(orient="records"),
            "relevant": relevant_df.to_dict(orient="records")
        }
    except MemoryError as me:
        logger.error(f"MemoryError in recommend_by_genre: {str(me)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=507, detail="Server out of memory. Try reducing top_n or check system resources.")
    except ValueError as ve:
        logger.error(f"ValueError in recommend_by_genre: {str(ve)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in recommend_by_genre: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
