# E-Cart Intelligent Product Recommendation & Demand Prediction

A comprehensive machine learning and AI-driven solution built for the E-Cart platform. This project provides personalized product recommendations, forecasts weekly inventory demand, segments customers using RFM analysis, and generates executive business insights using LLMs.

## Live Demo

![E-Cart Platform Demo](live_demo.gif)

---

## Key Features

* **Recommendation Engine:** Collaborative filtering model (Cosine Similarity) to suggest relevant products based on user interaction history.
* **Demand Prediction:** Time-series forecasting using a Random Forest Regressor to predict next week's product demand and prevent stockouts.
* **Customer Segmentation:** K-Means clustering to categorize users into actionable segments (e.g., High-Value Customers, Occasional Buyers) based on Recency, Frequency, and Monetary (RFM) metrics.
* **AI Business Insights:** Automated executive summaries and marketing strategies powered by Llama 3.1 via the Groq API.
* **Interactive Dashboard:** A clean, user-friendly Streamlit interface for exploring models, predictions, and reports.

---

## Project Structure

* `data/`: Contains raw dummy datasets and processed features ready for inference.
* `notebooks/`: Jupyter notebooks covering data cleaning, EDA, feature engineering, and model training.
* `models/`: Serialized machine learning model artifacts (`.pkl` files).
* `prompts/`: Text-based system prompts utilized for LLM generation.
* `app/`: Contains the Streamlit dashboard (`app.py`) and the connecting backend logic (`inference.py`).

---

## Getting Started

**1. Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt

```

**2. Environment Setup**
Create a `.env` file in the root directory and add your Groq API key to enable the AI insights and recommendation text generation:

```text
GROQ_API_KEY=your_groq_api_key_here

```

**3. Run the Application**
Navigate to the project root in your terminal and launch the Streamlit dashboard:

```bash
streamlit run app/app.py

```
