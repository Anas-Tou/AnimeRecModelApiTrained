import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz, process

class AnimeRecommendationModel:
    def __init__(self, load_from_artifacts=True):
        """
        Initialize the model without loading artifacts immediately.
        
        Parameters:
        - load_from_artifacts (bool): If True, set paths for artifacts to load later.
        """
        self.artifacts_loaded = False
        if load_from_artifacts:
            # Define paths for saved artifacts
            self.artifact_dir = "model_data"
            self.anime_df_path = os.path.join(self.artifact_dir, "anime_df.joblib")
            self.rating_df_path = os.path.join(self.artifact_dir, "rating_df.joblib")
            self.df_encoded_path = os.path.join(self.artifact_dir, "df_encoded.joblib")
            self.similarity_path = os.path.join(self.artifact_dir, "similarity.joblib")
            self.anime_idx_path = os.path.join(self.artifact_dir, "anime_idx.joblib")

            # Check if artifacts exist
            required_artifacts = [
                self.anime_df_path, self.rating_df_path, self.df_encoded_path,
                self.similarity_path, self.anime_idx_path
            ]
            missing_artifacts = [path for path in required_artifacts if not os.path.exists(path)]
            if missing_artifacts:
                raise FileNotFoundError(
                    f"Missing required artifacts: {missing_artifacts}. "
                    "Please run train_and_save() to generate these files."
                )

    def _load_artifacts(self):
        """Load artifacts into memory only when needed."""
        if not self.artifacts_loaded:
            self.anime_df = joblib.load(self.anime_df_path)
            self.rating_df = joblib.load(self.rating_df_path)
            self.df_encoded = joblib.load(self.df_encoded_path)
            self.similarity = joblib.load(self.similarity_path)
            self.anime_idx = joblib.load(self.anime_idx_path)
            self.artifacts_loaded = True

    @classmethod
    def train_and_save(cls, anime_csv_path, rating_csv_path, artifact_dir="model_data"):
        """
        Train the model on the provided data and save the artifacts for later use.
        
        Parameters:
        - anime_csv_path (str): Path to the anime CSV file.
        - rating_csv_path (str): Path to the ratings CSV file.
        - artifact_dir (str): Directory to save the artifacts (default: 'model_data').
        """
        # Create artifact directory if it doesn't exist
        if not os.path.exists(artifact_dir):
            os.makedirs(artifact_dir)

        # Load data
        anime_df = pd.read_csv(anime_csv_path)
        rating_df = pd.read_csv(rating_csv_path)

        # Create a temporary instance to access preprocessing methods
        temp_instance = cls.__new__(cls)
        object.__init__(temp_instance)

        # Assign data to the instance
        temp_instance.anime_df = anime_df
        temp_instance.rating_df = rating_df

        # Preprocess data with filtering
        temp_instance._preprocess_data()

        # Compute encoded features and similarity
        temp_instance.df_encoded = temp_instance._preprocessing(temp_instance.anime_df)
        # Convert to sparse matrix
        temp_instance.df_encoded = csr_matrix(temp_instance.df_encoded.values.astype('float32'))
        # Compute similarity as a sparse matrix
        temp_instance.similarity = cosine_similarity(temp_instance.df_encoded, dense_output=False)
        temp_instance.anime_idx = pd.Series(temp_instance.anime_df.index, index=temp_instance.anime_df['anime_id'])

        # Save artifacts with compression
        joblib.dump(temp_instance.anime_df, os.path.join(artifact_dir, "anime_df.joblib"), compress=3)
        joblib.dump(temp_instance.rating_df, os.path.join(artifact_dir, "rating_df.joblib"), compress=3)
        joblib.dump(temp_instance.df_encoded, os.path.join(artifact_dir, "df_encoded.joblib"), compress=3)
        joblib.dump(temp_instance.similarity, os.path.join(artifact_dir, "similarity.joblib"), compress=3)
        joblib.dump(temp_instance.anime_idx, os.path.join(artifact_dir, "anime_idx.joblib"), compress=3)

        print(f"Model trained and artifacts saved to {artifact_dir}")

    def recommend_similar_anime(self, anime_name, top_n=5, rating_threshold=6.0):
        self._load_artifacts()  # Load artifacts only when needed
        try:
            required_columns = ['name', 'genre', 'rating']
            if not all(col in self.anime_df.columns for col in required_columns):
                raise ValueError("DataFrame must contain 'name', 'genre', and 'rating' columns")

            # Normalize anime names for comparison
            def normalize_name(name):
                if not isinstance(name, str):
                    return ""
                # Remove special characters and normalize
                name = (name.replace(' ', '')
                        .replace(':', '')
                        .replace('{', '')
                        .replace('}', '')
                        .replace(',', '')
                        .replace('.', '')
                        .replace(';', '')
                        .replace('&', '')
                        .replace('#', '')
                        .replace("'", '')
                        .replace("°", '')
                        .lower()
                        .strip())
                return name

            anime_df = self.anime_df.copy()
            anime_df['genre'] = anime_df['genre'].fillna('').astype(str)
            anime_df['name_normalized'] = anime_df['name'].apply(normalize_name)
            normalized_input = normalize_name(anime_name)

            # Use fuzzy matching to find the closest match with higher confidence
            anime_names = anime_df['name_normalized'].tolist()
            match = process.extractOne(normalized_input, anime_names, scorer=fuzz.token_sort_ratio)
            if match is None or match[1] < 85:  # Increased to 85% for better accuracy
                raise ValueError(f"Anime '{anime_name}' not found in the dataset with sufficient similarity")

            matched_name_normalized = match[0]
            matched_anime_idx = anime_df[anime_df['name_normalized'] == matched_name_normalized].index[0]

            # Process genres for similarity
            def split_genres(text):
                if not text or not isinstance(text, str):
                    return []
                genres = [genre.strip() for genre in text.split(',') if genre.strip()]
                return genres

            anime_df['genre_list'] = anime_df['genre'].apply(split_genres)

            # Create a binary genre presence matrix for precise similarity
            all_genres = set()
            for genres in anime_df['genre_list']:
                all_genres.update(genres)
            genre_matrix = pd.DataFrame(0, index=anime_df.index, columns=list(all_genres))
            for idx, genres in anime_df['genre_list'].items():
                for genre in genres:
                    genre_matrix.at[idx, genre] = 1

            # Compute Jaccard similarity based on genre presence
            input_genres = set(anime_df.loc[matched_anime_idx, 'genre_list'])
            similarities = []
            for idx in anime_df.index:
                if idx == matched_anime_idx:
                    continue
                target_genres = set(anime_df.loc[idx, 'genre_list'])
                intersection = len(input_genres & target_genres)
                union = len(input_genres | target_genres)
                jaccard = intersection / union if union > 0 else 0
                similarities.append((idx, jaccard))

            # Sort by Jaccard similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:top_n * 2]]  # Fetch more to filter by rating
            recommended_df = anime_df.loc[top_indices].copy()

            # Apply rating threshold and sort by similarity
            recommended_df = recommended_df[recommended_df['rating'] >= rating_threshold]
            recommended_df = recommended_df.sort_values(by='rating', ascending=False)  # Optional: Keep top-rated
            recommendations = recommended_df[['name', 'genre', 'rating']].head(top_n)

            # Clean up temporary columns
            anime_df = anime_df.drop(columns=['name_normalized', 'genre_list'], errors='ignore')

            if recommendations.empty:
                return pd.DataFrame(columns=['name', 'genre', 'rating'])

            return recommendations

        except ValueError as e:
            if "not found in the dataset" in str(e):
                print("Anime Not Found")
            else:
                print(f"Error: {e}")
            return pd.DataFrame(columns=['name', 'genre', 'rating'])

    def _preprocess_data(self):
        self.anime_df = self.anime_df.dropna(subset=['genre', 'type']).reset_index(drop=True)
        self.anime_df = self.anime_df.drop_duplicates(subset=['name']).reset_index(drop=True)
        self.rating_df = self.rating_df.drop_duplicates().reset_index(drop=True)
        
        self.anime_df['episodes'] = self.anime_df['episodes'].replace(['Unknown', 'UNKNOWN', 'unknown', '-', ''], 0)
        self.anime_df['episodes'] = self.anime_df['episodes'].astype('int32')
        
        self.rating_df = self.rating_df[self.rating_df['rating'] != -1]
        
        # Filter ratings: keep only anime with at least 50 ratings
        anime_rating_counts = self.rating_df['anime_id'].value_counts()
        popular_anime = anime_rating_counts[anime_rating_counts >= 50].index
        self.rating_df = self.rating_df[self.rating_df['anime_id'].isin(popular_anime)]
        
        # Filter anime_df to match
        self.anime_df = self.anime_df[self.anime_df['anime_id'].isin(self.rating_df['anime_id'].unique())]
        
        # Downcast numeric columns
        self.anime_df['rating'] = self.anime_df['rating'].fillna(0).astype('float32')
        self.anime_df['members'] = self.anime_df['members'].fillna(0).astype('int32')
        self.anime_df['anime_id'] = self.anime_df['anime_id'].astype('int32')
        
        self.rating_df['rating'] = self.rating_df['rating'].astype('float32')
        self.rating_df['user_id'] = self.rating_df['user_id'].astype('int32')
        self.rating_df['anime_id'] = self.rating_df['anime_id'].astype('int32')

    def _preprocessing(self, anime_df):
        anime_df = anime_df.copy()
        # Reset index to ensure alignment
        anime_df = anime_df.reset_index(drop=True)
        
        def preprocess_genre(genre):
            genre_list = genre.split(',')
            processed_genres = [g.strip().replace(" ", "_") if len(g.strip().split()) > 1 else g.strip() for g in genre_list]
            return ', '.join(processed_genres)
        
        anime_df['genre_preprocessed'] = anime_df['genre'].apply(preprocess_genre)
        
        tfidf = TfidfVectorizer(encoding='utf-8', lowercase=True, stop_words=None, ngram_range=(1, 1))
        genre_encoded = tfidf.fit_transform(anime_df['genre_preprocessed'])
        features = tfidf.get_feature_names_out()
        # Ensure genre_encoded_df has the same index as anime_df
        genre_encoded_df = pd.DataFrame(
            data=genre_encoded.toarray(),
            columns=features,
            index=anime_df.index
        )
        
        type_encoded = pd.get_dummies(data=anime_df['type'])
        # Ensure type_encoded has the same index
        type_encoded = type_encoded.set_index(anime_df.index)
        
        df_encode = pd.concat((genre_encoded_df, type_encoded), axis=1)
        # Ensure df_encode has the correct number of rows
        df_encode = df_encode.loc[anime_df.index]
        
        df_encode['rating'] = anime_df['rating'].astype('float32')
        
        norm = StandardScaler()
        members_normalized = norm.fit_transform(anime_df[['members']].fillna(0))
        df_encode['members'] = members_normalized.flatten().astype('float32')
        
        # Convert all columns to float32 to ensure compatibility with csr_matrix
        df_encode = df_encode.astype('float32')
        
        return df_encode

    def predict_cb(self, user_id, anime_id):
        self._load_artifacts()  # Load artifacts only when needed
        watched_anime = self.rating_df[self.rating_df['user_id'] == user_id][['anime_id', 'rating']]
        watched_anime = watched_anime.sort_values(by='rating', ascending=False)[:20]
        
        target_anime = self.anime_idx[anime_id]
        sim_scores = []
        for watched in watched_anime['anime_id']:
            watched_idx = self.anime_idx[watched]
            # Access sparse matrix elements
            predicted_scores = self.similarity[target_anime, watched_idx]
            sim_scores.append(predicted_scores)
        
        return np.mean(sim_scores) if sim_scores else 0

    def get_recommendation(self, user_id, top_n=5):
        self._load_artifacts()  # Load artifacts only when needed
        all_anime = set(self.anime_df['anime_id'])
        watched_anime = set(self.rating_df[self.rating_df['user_id'] == user_id]['anime_id'])
        unwatched_anime = all_anime - watched_anime
        
        recommendation = []
        for anime_id in unwatched_anime:
            cb_score = self.predict_cb(user_id, anime_id)
            recommendation.append((anime_id, cb_score))
        
        recommendation = sorted(recommendation, key=lambda x: x[1], reverse=True)[:top_n]
        
        anime_ids = [i[0] for i in recommendation]
        predicted_ratings = [i[1] for i in recommendation]
        
        result_df = pd.DataFrame({'anime_id': anime_ids, 'predicted_rating': predicted_ratings})
        return pd.merge(self.anime_df, result_df, how='right', on='anime_id')[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members', 'predicted_rating']]

    def recommendation_by_genre(self, selected_genres, type_anime='all', top_n=5):
        self._load_artifacts()  # Load artifacts only when needed
        def filter_anime_by_genre(selected_genre, df):
            return df[df['genre'].apply(lambda x: any(genre in x.split(',') for genre in selected_genre))].reset_index(drop=True)

        def special_preprocessing(df):
            df = df.copy()
            
            def preprocess_genre(genre):
                genre_list = genre.split(',')
                processed_genres = [g.strip().replace(" ", "_") if len(g.strip().split()) > 1 else g.strip() for g in genre_list]
                return ', '.join(processed_genres)
            
            df['genre_preprocessed'] = df['genre'].apply(preprocess_genre)
            
            tfidf = TfidfVectorizer(encoding='utf-8', lowercase=True, stop_words=None, ngram_range=(1, 1))
            genre_encoded = tfidf.fit_transform(df['genre_preprocessed'])
            features = tfidf.get_feature_names_out()
            df_encode = pd.DataFrame(data=genre_encoded.toarray(), columns=features)
            df_encode['members'] = df['members'].fillna(0).astype('float32')
            return df_encode

        if type_anime is None:
            raise ValueError("Type can't be None!")
        
        filtered_df = filter_anime_by_genre(selected_genres, self.anime_df)
        df_encoded = special_preprocessing(filtered_df)
        similarity_matrix = cosine_similarity(df_encoded)
        
        relevant_genres = set(selected_genres)
        
        def count_irrelevant_genres(genre_string):
            anime_genres = set(genre_string.split(','))
            return len(anime_genres - relevant_genres)
        
        def count_relevant_genres(genre_string):
            anime_genres = set(genre_string.split(','))
            return len(anime_genres & relevant_genres)
        
        filtered_df["irrelevant_genre_count"] = filtered_df["genre"].apply(count_irrelevant_genres)
        filtered_df['relevant_genre'] = filtered_df['genre'].apply(count_relevant_genres)
        
        avg_similarity = similarity_matrix.mean(axis=0)
        avg_similarity /= np.where(filtered_df["irrelevant_genre_count"] > 0, filtered_df["irrelevant_genre_count"], 1)
        filtered_df['avg_similarity'] = avg_similarity
        
        relevant_df = filtered_df.sort_values(by=['relevant_genre', 'avg_similarity', 'members'], ascending=[False, False, False])
        popular_df = filtered_df.sort_values(by=['relevant_genre', 'members', 'avg_similarity'], ascending=[False, False, False])
        
        type_anime = type_anime.lower()
        if type_anime != 'all':
            relevant_df = relevant_df[relevant_df['type'].str.lower() == type_anime]
            popular_df = popular_df[popular_df['type'].str.lower() == type_anime]
            if relevant_df.empty:
                raise ValueError(f"No anime found for type '{type_anime}'!")
        
        return popular_df[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']].head(top_n), \
               relevant_df[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']].head(top_n)

    def __del__(self):
        pass