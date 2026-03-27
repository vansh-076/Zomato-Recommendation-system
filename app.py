import os
import streamlit as st
from Recommendation_system import RecommendationSystem


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR,"dataset", "final_dataset.csv")

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found: {DATA_PATH}\nRun your preprocessing step to create final_dataset.csv"
    )
    st.stop()

rs = RecommendationSystem()
try:
    rs.load_data(DATA_PATH)
    rs.model_develop()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                    url("https://images.unsplash.com/photo-1504674900247-0877df9cc836?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }   """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 300px;
    background-color:#ff0000; 
    padding:5px;
    border-radius:10px; 
    text-align:center;
    text-decoration: none;
">
    <h1 style="color: white; margin: 0;">Zomato</h1>
</div>
""",unsafe_allow_html=True)

st.title('Restaurant Recommendation System 🍽️',text_alignment='center')

st.header('Find Your Perfect Dining Spot in India with Zomato!',text_alignment='center')

st.markdown(
    """
    <h4 style="text-align:center; color: #fffff1;opacity: 2.0;">
    Discover the best restaurants in India based on your preferences.
    <br>Whether you are craving
    North Indian, South Indian, Chinese, <br>or any other cuisine, our recommendation system has got you covered!'
    </h4>
""",unsafe_allow_html=True
)
st.divider(width="stretch")
st.markdown("### Enter Your Preferences Below :",)
st.write("""<h5>This app recommends restaurants in the India region based on your preferences.</h5>""",unsafe_allow_html=True)

all_restaurants = sorted(rs.dataset['Restaurant Name'].dropna().unique())

restaurant_name = st.selectbox(
    "Restaurant Name (Optional)",
    options=[""] + all_restaurants,
    index=0,
    help=(
        "Select a restaurant name from the list (searchable), or leave blank to use cuisine filters. "
        "We handle multiple branches automatically when you choose a name."
    ),
)
all_cuisines=sorted(set(
    cuisines for row in rs.dataset['Cuisines'] for cuisines in row
))

preferred_cuisine=st.multiselect(
    "Preferred Cuisines (Optional)",
    options=all_cuisines,
    max_selections=5
)
rating_checker=st.checkbox(
    "Filter by Minimum Rating",
    value=False,
    help="Check this box to filter the recommendations based on a minimum rating threshold."
)
if rating_checker:
    min_rating=st.slider(
        "Minimum Rating (Optional)",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )

cities=sorted(rs.dataset['City'].unique())
selected_city=st.selectbox(
    "Select City (Optional)",
    options=['All'] + list(cities),
    index=0
)
city_filter= None if selected_city == 'All' else selected_city

n_recs=st.number_input(
    "Number of Recommendations",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)

show_only_online = st.toggle(
    "Show Only Online Delivery Restaurants",
    value=True,
    help="Check this box to filter the recommendations to only include restaurants that offer online delivery."
)

if st.button('Find Recommendations', type='primary',use_container_width=True):
    with st.spinner('Finding best matches....'):
        if preferred_restaurant_input :=restaurant_name.strip():
            if not preferred_cuisine and rating_checker==False:
                recs=rs.get_recommendation(
                    name=preferred_restaurant_input,
                    n=n_recs,
                    city=city_filter
                )
            elif not preferred_cuisine and rating_checker==True:
                recs=rs.recommendation_by_cuisines(
                    preferred_restaurant=preferred_restaurant_input,
                    n_recommendations=n_recs,
                    min_rating=min_rating,
                    city=city_filter
                )
            else:
                recs=rs.recommendation_by_cuisines(
                    preferred_restaurant=preferred_restaurant_input,
                    n_recommendations=n_recs,
                    min_rating=min_rating,
                    city=city_filter,
                    preferred_cuisines=preferred_cuisine
                )
        else:
            if not rating_checker:
               min_rating=0.0
            
            recs=rs.recommendation_by_cuisines(
                n_recommendations=n_recs,
                min_rating=min_rating,
                city=city_filter,
                preferred_cuisines=preferred_cuisine
            )
    if not recs.empty:
        if "message" in recs.columns:
            st.warning(recs.loc[0, "message"])
        else:
            st.success(f"Found {len(recs)} great recommendations!")

            fmt = {"Aggregate rating": "{:.1f} ⭐"}
            if "similarity_score" in recs.columns:
                fmt["similarity_score"] = "{:.0%}"

            st.dataframe(
                recs.style.format(fmt),
                use_container_width=True,
                hide_index=True,
            )
            
            cols = st.columns(3)
            for idx, row in recs.iterrows():
                with cols[idx % 3]:
                    st.markdown(f"**{row['Restaurant Name']}**")
                    st.caption(f"{row['City']} . {row['Cuisines']}")
                    st.metric("Rating", f"{row['Aggregate rating']:.1f} ⭐")
                    if "similarity_score" in row:
                        st.progress(row['similarity_score'])
                        st.caption(f"Similarity: {row['similarity_score']:.0%}")
                    st.divider()
    else:
        st.warning("No restaurants found matching your criteria. Try relaxing the filters.")
