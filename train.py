# train.py
from recommendation_model import AnimeRecommendationModel

anime_csv_path = "data/anime.csv"
rating_csv_path = "data/rating.csv"

AnimeRecommendationModel.train_and_save(anime_csv_path, rating_csv_path)