import pandas as pd
import numpy as np
import pickle
import os
from groq import Groq
from dotenv import load_dotenv 
load_dotenv() 

#get the absolute path of the project root directory (one level up from app/)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#initialize groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model_name = "llama-3.1-8b-instant"

#load global datasets using dynamic absolute paths
products_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'cleaned_products.csv'))

#load machine learning models and matrices
with open(os.path.join(base_dir, 'models', 'user_item_matrix.pkl'), 'rb') as f:
    user_item_matrix = pickle.load(f)

with open(os.path.join(base_dir, 'models', 'user_similarity_matrix.pkl'), 'rb') as f:
    user_similarity_df = pickle.load(f)

with open(os.path.join(base_dir, 'models', 'demand_rf_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

#helper function to read system prompts
def load_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

#recommendation logic
def get_ml_recommendations(user_id, num_recs=3):
    #check if user exists in our matrix
    if user_id not in user_item_matrix.index:
        return []
    
    #get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10).index
    target_user_history = user_item_matrix.loc[user_id]
    items_already_seen = target_user_history[target_user_history > 0].index.tolist()
    
    recommendation_scores = {}
    for similar_user in similar_users:
        similarity_score = user_similarity_df.loc[user_id, similar_user]
        similar_user_history = user_item_matrix.loc[similar_user]
        items_interacted = similar_user_history[similar_user_history > 0].index
        
        for item in items_interacted:
            if item not in items_already_seen:
                if item not in recommendation_scores:
                    recommendation_scores[item] = 0
                #accumulate weighted score
                recommendation_scores[item] += similar_user_history[item] * similarity_score
                
    #sort and filter top n
    sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    top_n_ids = [item_id for item_id, score in sorted_recs[:num_recs]]
    
    #map back to product details
    recs_df = products_df[products_df['product_id'].isin(top_n_ids)]
    return recs_df[['product_name', 'category', 'brand']].to_dict('records')

def generate_llm_recommendation_text(user_id):
    #get raw ml outputs
    raw_recs = get_ml_recommendations(user_id)
    if not raw_recs:
        return "user not found or not enough interaction data to generate targeted recommendations."
    
    #prepare payload for the llm
    payload = f"User ID: {user_id}\nRecommendations: {raw_recs}"
    prompt_path = os.path.join(base_dir, 'prompts', 'recommendation_prompt.txt')
    system_prompt = load_prompt(prompt_path)
    
    #call groq api with llama 3.1
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload}
        ],
        model=model_name,
        temperature=0.3
    )
    return response.choices[0].message.content

#demand prediction logic
def get_demand_predictions():
    #load the latest features needed for next week's forecast
    latest_data_path = os.path.join(base_dir, 'data', 'processed', 'latest_demand_features.csv')
    latest_data = pd.read_csv(latest_data_path)
    
    #shift features forward to simulate the future week
    latest_data['week'] = latest_data['week'] + 1
    
    x_future = latest_data[['product_id', 'week', 'lag_1', 'lag_2']]
    
    #predict using the loaded random forest model
    preds = rf_model.predict(x_future)
    latest_data['predicted_demand'] = np.ceil(preds).astype(int)
    
    #merge to get product names and return the top 15 highest demand items
    output_df = latest_data.merge(products_df, on='product_id', how='left')
    top_demand = output_df[['product_name', 'predicted_demand']].sort_values(by='predicted_demand', ascending=False)
    
    return top_demand.head(15)

#business insights logic
def generate_business_insights():
    #gather data context to feed the llm
    demand_preds = get_demand_predictions().to_dict('records')
    
    #load pre-calculated customer segment summaries
    segments_path = os.path.join(base_dir, 'data', 'processed', 'final_customer_segments.csv')
    segments_df = pd.read_csv(segments_path)
    segment_counts = segments_df['customer_segment'].value_counts().to_dict()
    
    #construct the data payload
    context = f"Next Week Highest Demand Predictions: {demand_preds}\nCustomer Base Breakdown: {segment_counts}"
    prompt_path = os.path.join(base_dir, 'prompts', 'insights_prompt.txt')
    system_prompt = load_prompt(prompt_path)
    
    #call groq api for the executive report
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        model=model_name,
        temperature=0.4
    )
    return response.choices[0].message.content