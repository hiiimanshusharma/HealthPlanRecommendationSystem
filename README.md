# HealthPlanRecommendationSystem

I've deployed a Streamlit web application for health plan recommendations, and it's been quite an exciting project. With this app, users can simply input their symptoms or test reports, and with the click of a button, it predicts their medical condition based on the provided text. The application then suggests the top drugs for that condition,based on trained model tfidfvectorizer.pkl,  passmodel.pkl considering factors like high ratings and usefulness counts from a dataset, trained in Medicine_Recommendation_System.ipynb file using Passive Agression Classifier model with  TFIDF vectorization. What's really fascinating is that it doesn't stop there – it also generates a personalized health plan using a GPT-2 language model. This tool has the potential to offer valuable health insights and recommendations to users, making it a useful addition to the healthcare tech landscape. 

```$ pip install -r requirements.txt```

```$ streamlit run st_app.py```

Demonstration link : https://www.loom.com/share/480f37c8501e401ab7a208d27bb5ff61?sid=f99be8ed-df28-40c0-b22a-39e1ea122cf9
