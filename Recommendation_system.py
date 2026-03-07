import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationSystem:
    """A lightweight restaurant recommendation system based on cuisine similarity."""

    def __init__(self):
        self.sim_matrix = None
        self.dataset = None
        self.tfidf_matrix = None
        self.index = None

    def load_data(self, file_path: str):
        """Load and preprocess the dataset."""
        df = pd.read_csv(file_path, encoding="latin-1")
        df["Cuisines"] = df["Cuisines"].str.split(",").apply(
            lambda x: [val.strip().replace(" ", "_") for val in x]
        )
        df["Cuisines_processed"] = df["Cuisines"].apply(lambda x: " ".join(x))

        df["Combined_col"] = (
            df["Restaurant Name"].str.lower()
            + " "
            + df["City"].str.lower()
            + " "
            + df["Locality"].str.lower()
            + " "
            + df["Cuisines_processed"].str.lower()
        )

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        self.dataset = df
        return self.dataset

    def model_develop(self):
        """Build the TF-IDF model and similarity matrix."""
        if self.dataset is None:
            raise ValueError("Load the data first using the load_data() method.")

        df = self.dataset
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )
        vectors = tfidf.fit_transform(df["Combined_col"])
        similarity_matrix = cosine_similarity(vectors, vectors)

        restaurant_idx = dict(zip(df["Restaurant ID"], df.index))
        self.sim_matrix = similarity_matrix
        self.index = restaurant_idx
        self.tfidf_matrix = vectors

    def name_to_id(self, restaurant_name: str, city_hint: str | bool = False):
        """Resolve a restaurant name to its internal restaurant ID."""
        match = self.dataset[self.dataset["Restaurant Name"].str.lower() == restaurant_name.lower()]
        if match.empty:
            return None
        if len(match) == 1:
            return match.iloc[0]["Restaurant ID"]

        if city_hint:
            match = match[match["City"].str.lower().str.contains(str(city_hint).lower())]
            if len(match) == 1:
                return match.iloc[0]["Restaurant ID"]
            if len(match) == 0:
                return None

        match = match.sort_values(by=["Aggregate rating", "Votes"], ascending=[False, False])
        self.print_restaurant_branch(match.head(5))
        return match.iloc[0]["Restaurant ID"]

    def print_restaurant_branch(self, restaurant_branch: pd.DataFrame):
        columns = [
            "Restaurant Name",
            "Cuisines_processed",
            "Locality",
            "City",
            "Has Table booking",
            "Has Online delivery",
            "Average Cost for two",
            "Aggregate rating",
        ]
        print(f"Printing the top 5 branches for the restaurant {restaurant_branch.iloc[0]['Restaurant Name']}")
        print(restaurant_branch[columns].to_string(index=False))
        print("=" * 200)

    def get_recommendation(self, name: str, n: int, city: str):
        result = self.name_to_id(name, city)
        if result is None:
            # Always return a DataFrame so the caller can handle empty results cleanly
            return pd.DataFrame(
                [{"Restaurant Name": name, "message": "Restaurant not available."}]
            )

        rest_idx = self.index[result]
        score = list(enumerate(self.sim_matrix[rest_idx]))
        sort_score = sorted(score, key=lambda x: x[1], reverse=True)
        top_n = sort_score[1 : n + 1]
        top_index = [i[0] for i in top_n]

        recommend = self.dataset.iloc[top_index].copy()
        recommend["similarity_score"] = [i[1] for i in top_n]
        return recommend[
            [
                "Restaurant Name",
                "City",
                "Locality",
                "Cuisines_processed",
                "Aggregate rating",
                "Votes",
                "similarity_score",
            ]
        ]

    def recommendation_by_cuisines(
        self,
        preferred_cuisines: list[str],
        n_recommendations: int,
        min_rating: float | None = None,
        city: str | None = None,
        preferred_restaurant: str | None = None,
    ):
        user_cuisines = [cuisine.strip().replace(" ", "_").lower() for cuisine in preferred_cuisines]

        def Has_preferred_cuisine(row):
            if pd.isna(row["Cuisines_processed"]):
                return False
            restaurant_cuisine = [c.lower().strip() for c in row["Cuisines_processed"].split(" ")]
            return any(p in restaurant_cuisine for p in user_cuisines)

        filtered_data = self.dataset[self.dataset.apply(Has_preferred_cuisine, axis=1)].copy()

        if city:
            filtered_data = filtered_data[
                filtered_data["City"].str.lower().str.strip().str.contains(city.lower())
            ]
        if min_rating is not None:
            filtered_data = filtered_data[filtered_data["Aggregate rating"] >= min_rating]

        if filtered_data.shape[0] == 0:
            return pd.DataFrame()

        if preferred_restaurant:
            rest_id = self.name_to_id(preferred_restaurant, city)
            if not rest_id:
                return pd.DataFrame()

            idx = self.index[rest_id]
            sim_score = self.sim_matrix[idx]
            filter_indices = filtered_data.index
            filtered_sim_score = sim_score[filter_indices]
            sorted_idx = np.argsort(filtered_sim_score)[::-1]
            top_indices = filter_indices[sorted_idx][:n_recommendations]

            result = self.dataset.loc[top_indices].copy()
            result["similarity_score"] = filtered_sim_score[sorted_idx][:n_recommendations]
        else:
            result = (
                filtered_data.sort_values(by=["Votes", "Aggregate rating"], ascending=[False, False])
                .head(n_recommendations)
            )

        columns = [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Aggregate rating",
            "Votes",
            "Price range",
        ]
        if "similarity_score" in result.columns:
            columns.append("similarity_score")

        return result[columns].reset_index(drop=True)
