import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class UserRecommendations(Base):
    __tablename__ = 'user_recommendations'

    customer_id = Column(String, ForeignKey('customer_profiles.customer_id'), primary_key=True)
    recommendations = Column(JSON)
    last_updated = Column(DateTime)

    customer = relationship("CustomerProfile", back_populates="recommendations")

class CustomerProfile(Base):
    __tablename__ = 'customer_profiles'

    customer_id = Column(String, primary_key=True)
    full_name = Column(String)
    email = Column(String, unique=True)
    username = Column(String, unique=True)
    phone_number = Column(String)
    age = Column(Integer)
    gender = Column(String)
    location = Column(String)

    recommendations = relationship("UserRecommendations", back_populates="customer", uselist=False)

class Product(Base):
    __tablename__ = 'products'

    product_id = Column(Integer, primary_key=True)
    category = Column(String)
    price = Column(Float)
    brand = Column(String)
    rating = Column(Float)
    popularity = Column(Float)
    last_updated = Column(DateTime)

class EcommerceRecommendationSystem:
    def __init__(self, db_url='sqlite:///recommendations.db'):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        self.user_model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
        self.product_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.user_scaler = StandardScaler()
        self.product_scaler = StandardScaler()

        self.customer_data = None
        self.product_data = None
        self.unique_categories = None
        self.category_weights = defaultdict(float)

        self._load_data()
        self._train_models()

    def _load_data(self):
        try:
            self.customer_data = pd.read_csv('customer_data/customer_data_collection.csv')
            self.customer_data['Browsing_History'] = self.customer_data['Browsing_History'].apply(eval)
            self.customer_data['Purchase_History'] = self.customer_data['Purchase_History'].apply(eval)

            self.product_data = pd.read_csv('product_data/product_recommendation_data.csv')

            try:
                db_products = pd.read_sql('SELECT * FROM products', self.engine)
                if not db_products.empty:
                    self.product_data = pd.concat([self.product_data, db_products.rename(columns={
                        'category': 'Category',
                        'price': 'Price',
                        'brand': 'Brand',
                        'rating': 'Product_Rating',
                        'popularity': 'Probability_of_Recommendation',
                        'product_id': 'Product_ID'
                    })], ignore_index=True)
            except Exception as e:
                logger.warning(f"Could not load products from database: {str(e)}")

            self.product_data['Price'] = pd.to_numeric(self.product_data['Price'], errors='coerce')
            self.product_data['Product_Rating'] = pd.to_numeric(self.product_data['Product_Rating'], errors='coerce')
            self.product_data['Probability_of_Recommendation'] = pd.to_numeric(
                self.product_data['Probability_of_Recommendation'], errors='coerce')

            self.product_data = self.product_data.assign(
                Price=self.product_data['Price'].fillna(self.product_data['Price'].mean()),
                Product_Rating=self.product_data['Product_Rating'].fillna(self.product_data['Product_Rating'].mean()),
                Probability_of_Recommendation=self.product_data['Probability_of_Recommendation'].fillna(
                    self.product_data['Probability_of_Recommendation'].mean())
            )

            self.unique_categories = sorted(set(
                cat for _, row in self.customer_data.iterrows()
                for cat in row['Browsing_History'] + row['Purchase_History']
            ))

            logger.info("Data loaded and preprocessed successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _train_models(self):
        try:
            user_features = []
            for _, row in self.customer_data.iterrows():
                feature_vector = []
                for cat in self.unique_categories:
                    browse_count = row['Browsing_History'].count(cat)
                    purchase_count = row['Purchase_History'].count(cat)
                    feature_vector.append(browse_count * 0.3 + purchase_count * 0.7)
                user_features.append(feature_vector)

            user_features = np.array(user_features)
            user_features_scaled = self.user_scaler.fit_transform(user_features)
            self.user_model.fit(user_features_scaled)

            product_features = self.product_data[['Price', 'Product_Rating', 'Probability_of_Recommendation']].values
            product_features_scaled = self.product_scaler.fit_transform(product_features)
            self.product_model.fit(product_features_scaled)

            logger.info("Models trained successfully")
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def _calculate_category_weights(self, browsing_history: list, purchase_history: list) -> dict:
        weights = defaultdict(float)
        time_decay = 0.9

        for i, category in enumerate(reversed(purchase_history)):
            weights[category] += 0.7 * (time_decay ** i)

        for i, category in enumerate(reversed(browsing_history)):
            weights[category] += 0.3 * (time_decay ** i)

        category_popularity = self._calculate_category_popularity()
        for category in weights:
            weights[category] *= (1 + (1 - category_popularity.get(category, 0)))

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _calculate_category_popularity(self) -> dict:
        popularity = defaultdict(float)
        for _, row in self.customer_data.iterrows():
            for cat in row['Browsing_History']:
                popularity[cat] += 0.3
            for cat in row['Purchase_History']:
                popularity[cat] += 0.7

        max_popularity = max(popularity.values()) if popularity else 1
        return {k: v / max_popularity for k, v in popularity.items()}

    def get_recommendations(self, user_id: str, browsing_history: list, purchase_history: list) -> list:
        try:
            existing_recs = self.session.query(
                UserRecommendations.recommendations,
                UserRecommendations.last_updated
            ).filter_by(customer_id=user_id).first()

            if existing_recs:
                if (datetime.now() - existing_recs.last_updated).total_seconds() < 86400:
                    logger.info(f"Using cached recommendations for user {user_id}")
                    cached_recs = existing_recs.recommendations
                    if isinstance(cached_recs, str):
                        cached_recs = json.loads(cached_recs)
                    return cached_recs

            user_categories = list(set(browsing_history + purchase_history))
            logger.info(f"User categories: {user_categories}")

            user_products = self.product_data[self.product_data['Category'].isin(user_categories)]

            if len(user_products) == 0:
                logger.warning("No products found in user categories, using all products")
                user_products = self.product_data

            grouped_products = user_products.groupby('Category')
            recommendations = []

            for category in user_categories:
                if category in grouped_products.groups:
                    category_products = grouped_products.get_group(category)
                    top_products = category_products.nlargest(2, 'Product_Rating')
                    for _, product in top_products.iterrows():
                        recommendations.append({
                            'Category': product['Category'],
                            'Product_ID': product['Product_ID'],
                            'Price': float(product['Price']),
                            'Brand': product['Brand'],
                            'Product_Rating': float(product['Product_Rating']),
                            'score': float(product['Product_Rating'])
                        })

            if len(recommendations) < 10:
                remaining_products = self.product_data[~self.product_data['Category'].isin(user_categories)]
                remaining_products = remaining_products.nlargest(10 - len(recommendations), 'Product_Rating')
                for _, product in remaining_products.iterrows():
                    recommendations.append({
                        'Category': product['Category'],
                        'Product_ID': product['Product_ID'],
                        'Price': float(product['Price']),
                        'Brand': product['Brand'],
                        'Product_Rating': float(product['Product_Rating']),
                        'score': float(product['Product_Rating'])
                    })

            self._store_recommendations(user_id, recommendations)

            return recommendations[:10]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def _store_recommendations(self, user_id: str, recommendations: list):
  
        try:
            with self.Session() as session:
                 existing_recs = session.query(UserRecommendations).filter_by(customer_id=user_id).first()
                 if existing_recs:
                    existing_recs.recommendations = json.dumps(recommendations)
                    existing_recs.last_updated = datetime.now()
                 else:
                     new_recs = UserRecommendations(
                        customer_id=user_id,
                        recommendations=json.dumps(recommendations),
                        last_updated=datetime.now(),
                    )
                     session.add(new_recs)

                 session.commit()
                 logger.info(f"Stored recommendations for user {user_id}")
        except Exception as e:
             logger.error(f"Error storing recommendations: {str(e)}")
             raise


    def print_recommendations(self, customer_id: str, browsing_history: list, purchase_history: list):
        try:
            recommendations = self.get_recommendations(customer_id, browsing_history, purchase_history)

            print(f"\nRecommendations for User {customer_id}:")
            print("=" * 50)
            print(f"User's Browsing History: {browsing_history}")
            print(f"User's Purchase History: {purchase_history}")
            print("\nRecommended Products:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\nRecommendation {i}:")
                print(f"Product ID: {rec['Product_ID']}")
                print(f"Category: {rec['Category']}")
                print(f"Price: {rec['Price']}")
                print(f"Brand: {rec['Brand']}")
                print(f"Rating: {rec['Product_Rating']}")
                print(f"Recommendation Score: {rec['score']:.2f}")
            print("=" * 50)
        except Exception as e:
            logger.error(f"Error printing recommendations: {str(e)}")
            raise
