
import streamlit as st
import pickle

sdg_names = {1: 'No Poverty', 2: 'Zero Hunger', 3: 'Good Health and Well-being', 4: 'Quality Education', 5: 'Gender Equality', 6: 'Clean Water and Sanitation', 7: 'Affordable and Clean Energy', 8: 'Decent Work and Economic Growth', 9: 'Industry, Innovation and Infrastructure', 10: 'Reduced Inequalities', 11: 'Sustainable Cities and Communities', 12: 'Responsible Consumption and Production', 13: 'Climate Action', 14: 'Life Below Water', 15: 'Life on Land', 16: 'Peace, Justice and Strong Institutions', 17: 'Partnerships for the Goals'}

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("SDG-Based News Classification System")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
#         proba = max(model.predict_proba(input_vec)[0]) * 100
        
        st.success(f"Predicted SDG: {prediction} - {sdg_names[prediction]}")
#        st.info(f"Confidence: {proba:.2f}%")
    else:
        st.warning("Please enter news text.")
