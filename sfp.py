import streamlit as st
import pandas as pd
import pickle
import os
from scipy.stats import poisson
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import streamlit.components.v1 as components
import base64
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
# --- Custom CSS for sidebar radio ---
st.markdown("""
    <style>
    /* Hide the radio circle */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Force white text for all nested text elements in radio labels */
    section[data-testid="stSidebar"] div[role="radiogroup"] label * {
        color: white !important;
    }

    /* Style each radio option */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        border-radius: 25px;
        padding: 10px 20px;
        margin: 10px 0;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        box-sizing: border-box;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: linear-gradient(90deg, #8f94fb, #4e54c8);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Active (selected) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
    </style>
""", unsafe_allow_html=True)



# --- Sidebar Navigation ---
st.sidebar.title("⚽ Somali Predictor")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📜 History"],   # add emoji icons here
    label_visibility="collapsed"
)


# --- MongoDB Setup ---
MONGO_URI = "mongodb+srv://Abas:Abaas44@cluster0.ybytw.mongodb.net/somaliFootball?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["somaliFootball"]   # database
collection = db["prediction"]   # collection

# --- Save Prediction ---
def save_prediction_to_history(home_team, away_team, home_prob, draw_prob, away_prob, predicted_outcome):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "home_team": home_team,
        "away_team": away_team,
        "home_prob": round(home_prob, 1),
        "draw_prob": round(draw_prob, 1),
        "away_prob": round(away_prob, 1),
        "predicted_outcome": predicted_outcome,
    }

    collection.insert_one(entry)  # Save into MongoDB

# --- Load History ---
def load_prediction_history():
    history = []
    for doc in collection.find().sort("timestamp", 1):  # sort oldest → newest
        entry = {
            "_id": str(doc["_id"]),   # keep _id as string for buttons
            "timestamp": doc.get("timestamp"),
            "home_team": doc.get("home_team"),
            "away_team": doc.get("away_team"),
            "home_prob": doc.get("home_prob"),
            "draw_prob": doc.get("draw_prob"),
            "away_prob": doc.get("away_prob"),
            "predicted_outcome": doc.get("predicted_outcome"),
        }
        history.append(entry)
    return history
# --- Pages ---
if page == "🏠 Home":
        # Start the main content

    # Load df_team_strength from pickle
    try:
        with open('df_team_strength.pkl', 'rb') as file:
            df_team_strength = pickle.load(file)
    except FileNotFoundError:
        st.error("Could not find 'df_team_strength.pkl'. Please ensure the file exists in the script directory.")
        st.stop()

    # Ensure index is clean
    df_team_strength.index = df_team_strength.index.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Cleaned4_Somali_league_datasets.csv")

    try:
        df = pd.read_csv(file_path)
        print("✅ CSV Loaded Successfully!")
    except FileNotFoundError:
        st.error("Could not find 'Cleaned4_Somali_league_datasets.csv'. Please ensure the file exists in the script directory.")
        st.stop()

    # Load XGBoost model and scaler (replace random forest)
    try:
        with open('xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Could not load a required file: {e}. Please ensure 'xgboost_model.pkl' and 'scaler.pkl' are present.")
        st.stop()

    # Define feature columns - must match the exact order used during model training
    feature_columns = [
        'Home_Score', 'Away_Score', 'Home_Goal_Diff', 'Away_Goal_Diff', 
        'Home_Last_5_Wins', 'Away_Last_5_Wins', 'Home_Last_5_Points', 
        'Away_Last_5_Points', 'H2H_Home_Wins', 'H2H_Home_Losses', 
        'H2H_Away_Wins', 'H2H_Away_Losses', 'H2H_Draws',
        'H2H_Home_Win_Ratio', 'H2H_Away_Win_Ratio', 'Last_5_Point_Diff', 'H2H_Win_Diff'
    ]

    # Functions for validation and data processing
    def validate_teams(home_team, away_team, df):
        home_exists = home_team in df['Home_Team'].values or home_team in df['Away_Team'].values
        away_exists = away_team in df['Away_Team'].values or away_team in df['Home_Team'].values
        if not home_exists:
            print(f"Error: '{home_team}' is not a valid team in the Somali League dataset.")
            return False
        if not away_exists:
            print(f"Error: '{away_team}' is not a valid team in the Somali League dataset.")
            return False
        if home_team == away_team:
            print("Error: Home Team and Away Team cannot be the same.")
            return False
        return True

    def get_h2h_info(home_team, away_team, df):
        h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                        ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))]
        
        if h2h_matches.empty:
            return {'total_matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0}
        
        h2h_home = df[(df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)]
        if not h2h_home.empty:
            latest_h2h = h2h_home.iloc[-1]
            home_wins = int(latest_h2h['H2H_Home_Wins'])
            home_losses = int(latest_h2h['H2H_Home_Losses'])
            away_wins = int(latest_h2h['H2H_Away_Wins'])
            away_losses = int(latest_h2h['H2H_Away_Losses'])
            draws = int(latest_h2h['H2H_Draws'])
            total_matches = home_wins + home_losses + draws
        else:
            total_matches = len(h2h_matches)
            home_wins_as_home = len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 0)])
            away_wins_as_away = len(h2h_matches[(h2h_matches['Away_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
            draws = len(h2h_matches[h2h_matches['Result_Encoded'] == 1])
            home_wins = home_wins_as_home + len(h2h_matches[(h2h_matches['Home_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
            away_wins = away_wins_as_away + len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 2)])

        return {'total_matches': total_matches, 'home_wins': home_wins, 'away_wins': away_wins, 'draws': draws}

    def get_team_form(team, df):
        team_matches = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].sort_index(ascending=False).head(5)
        
        if team_matches.empty:
            return {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
        
        wins = 0
        draws = 0
        losses = 0
        points = 0
        
        for _, match in team_matches.iterrows():
            if match['Home_Team'] == team:
                if match['Result_Encoded'] == 0:  # Home win
                    wins += 1
                    points += 3
                elif match['Result_Encoded'] == 1:  # Draw
                    draws += 1
                    points += 1
                else:  # Away win (loss for home team)
                    losses += 1
            else:  # Team is Away_Team
                if match['Result_Encoded'] == 2:  # Away win
                    wins += 1
                    points += 3
                elif match['Result_Encoded'] == 1:  # Draw
                    draws += 1
                    points += 1
                else:  # Home win (loss for away team)
                    losses += 1
        
        return {'wins': wins, 'draws': draws, 'losses': losses, 'points': points}

    def get_last_5_matches_details(team, df):
        """Get detailed information about the last 5 matches for a team"""
        team_matches = df[(df['Home_Team'] == team) | (df['Away_Team'] == team)].sort_index(ascending=False).head(5)
        
        matches_details = []
        
        for _, match in team_matches.iterrows():
            if match['Home_Team'] == team:
                opponent = match['Away_Team']
                team_score = int(match['Home_Score'])
                opponent_score = int(match['Away_Score'])
                is_home = True
            else:
                opponent = match['Home_Team']
                team_score = int(match['Away_Score'])
                opponent_score = int(match['Home_Score'])
                is_home = False
            
            # Determine result
            if team_score > opponent_score:
                result = "W"
                result_class = "result-win"
            elif team_score < opponent_score:
                result = "L"
                result_class = "result-loss"
            else:
                result = "D"
                result_class = "result-draw"
            
            matches_details.append({
                'opponent': opponent,
                'team_score': team_score,
                'opponent_score': opponent_score,
                'result': result,
                'result_class': result_class,
                'is_home': is_home,
                'season': match['Season']
            })
        
        return matches_details

    def get_recent_match_result(home_team, away_team, df):
        h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                        ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))].sort_index(ascending=False)
        
        if h2h_matches.empty:
            return None
        
        latest_match = h2h_matches.iloc[0]
        home_team_match = latest_match['Home_Team']
        away_team_match = latest_match['Away_Team']
        home_score = int(latest_match['Home_Score'])
        away_score = int(latest_match['Away_Score'])
        season = latest_match['Season']
        
        return {'home_team': home_team_match, 'away_team': away_team_match, 'home_score': home_score, 'away_score': away_score, 'season': season}

    def prepare_input(home_team, away_team, df, feature_columns):
        home_data = df[df['Home_Team'] == home_team]
        away_data = df[df['Away_Team'] == away_team]
        
        if home_data.empty:
            home_data = df
            print(f"Warning: No historical data for {home_team} as Home Team. Using dataset averages.")
        if away_data.empty:
            away_data = df
            print(f"Warning: No historical data for {away_team} as Away Team. Using dataset averages.")
        
        home_means = home_data.mean(numeric_only=True)
        away_means = away_data.mean(numeric_only=True)
        
        # Get H2H information for additional features
        h2h_info = get_h2h_info(home_team, away_team, df)
        home_form = get_team_form(home_team, df)
        away_form = get_team_form(away_team, df)
        
        features_dict = {}
        for col in feature_columns:
            if col == 'H2H_Away_Win_Ratio':
                # Calculate H2H away win ratio
                total_h2h = h2h_info['total_matches']
                if total_h2h > 0:
                    features_dict[col] = h2h_info['away_wins'] / total_h2h
                else:
                    features_dict[col] = 0.0
            elif col == 'H2H_Home_Win_Ratio':
                # Calculate H2H home win ratio
                total_h2h = h2h_info['total_matches']
                if total_h2h > 0:
                    features_dict[col] = h2h_info['home_wins'] / total_h2h
                else:
                    features_dict[col] = 0.0
            elif col == 'H2H_Win_Diff':
                # Calculate H2H win difference (home wins - away wins)
                features_dict[col] = h2h_info['home_wins'] - h2h_info['away_wins']
            elif col == 'Last_5_Point_Diff':
                # Calculate point difference in last 5 matches
                features_dict[col] = home_form['points'] - away_form['points']
            elif 'Home' in col:
                features_dict[col] = home_means.get(col, df[col].mean()) if col in home_means else df[col].mean()
            elif 'Away' in col:
                features_dict[col] = away_means.get(col, df[col].mean()) if col in away_means else df[col].mean()
            else:
                features_dict[col] = (home_means.get(col, 0) + away_means.get(col, 0)) / 2 if col in home_means else df[col].mean()
        
        input_data = pd.DataFrame([features_dict], columns=feature_columns)
        input_data_scaled = scaler.transform(input_data)
        return input_data_scaled

    def predict_points(home, away):
        if home in df_team_strength.index and away in df_team_strength.index:
            lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
            lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']
            prob_home, prob_away, prob_draw = 0, 0, 0
            
            for x in range(0, 11):  # Number of goals home team
                for y in range(0, 11):  # Number of goals away team
                    p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                    if x == y:
                        prob_draw += p
                    elif x > y:
                        prob_home += p
                    else:
                        prob_away += p
            
            points_home = 3 * prob_home + prob_draw
            points_away = 3 * prob_away + prob_draw
            return points_home, points_away, prob_home, prob_draw, prob_away
        else:
            return 0, 0, 0, 0, 0

    def adjust_probabilities_by_h2h(home_team, away_team, df, home_prob, draw_prob, away_prob):
        # Get H2H information
        h2h_matches = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                        ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))]
        
        if h2h_matches.empty:
            return home_prob, draw_prob, away_prob
        
        # Calculate H2H wins for each team
        home_wins = len(h2h_matches[(h2h_matches['Home_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 0)]) + \
                    len(h2h_matches[(h2h_matches['Away_Team'] == home_team) & (h2h_matches['Result_Encoded'] == 2)])
        
        away_wins = len(h2h_matches[(h2h_matches['Home_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 0)]) + \
                    len(h2h_matches[(h2h_matches['Away_Team'] == away_team) & (h2h_matches['Result_Encoded'] == 2)])
        
        draws = len(h2h_matches[h2h_matches['Result_Encoded'] == 1])
        total_matches = len(h2h_matches)
        
        # Calculate H2H win percentages
        home_h2h_win_pct = (home_wins / total_matches) * 100 if total_matches > 0 else 0
        away_h2h_win_pct = (away_wins / total_matches) * 100 if total_matches > 0 else 0
        draw_h2h_pct = (draws / total_matches) * 100 if total_matches > 0 else 0
        
        # Adjust probabilities based on H2H record
        if home_wins > away_wins:
            # Home team has better H2H record
            home_prob = (home_prob * 0.7) + (home_h2h_win_pct * 0.3)
            away_prob = (away_prob * 0.7) + (away_h2h_win_pct * 0.3)
            draw_prob = (draw_prob * 0.7) + (draw_h2h_pct * 0.3)
        elif away_wins > home_wins:
            # Away team has better H2H record
            home_prob = (home_prob * 0.7) + (home_h2h_win_pct * 0.3)
            away_prob = (away_prob * 0.7) + (away_h2h_win_pct * 0.3)
            draw_prob = (draw_prob * 0.7) + (draw_h2h_pct * 0.3)
        else:
            # Equal H2H record, increase draw probability
            home_prob = (home_prob * 0.7) + (home_h2h_win_pct * 0.3)
            away_prob = (away_prob * 0.7) + (away_h2h_win_pct * 0.3)
            draw_prob = (draw_prob * 0.7) + (draw_h2h_pct * 0.3)
        
        # Normalize probabilities to sum to 100%
        total = home_prob + draw_prob + away_prob
        home_prob = (home_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_prob = (away_prob / total) * 100
        
        return home_prob, draw_prob, away_prob

    # Function to get team logo path
    def get_team_logo(team_name):
        logo_folder = "TeamLogo"
        
        
        
        for ext in ["png", "jpeg", "jpg"]:
            logo_path = os.path.join(logo_folder, f"{team_name}.{ext}")
            if os.path.exists(logo_path):
                return logo_path
        
        return None

    # Function to create circular logos
    def create_circular_logo(image_path, size=(150, 150)):
        image = Image.open(image_path).convert("RGBA")
        image = image.resize(size, Image.LANCZOS)
        
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        
        circular_image = Image.new("RGBA", size, (255, 255, 255, 0))
        circular_image.paste(image, (0, 0), mask)
        
        return circular_image

    # Define Seria A teams (make sure names match those in df_team_strength.index)
    SERIA_A_TEAMS = [
        "Horseed SC",
        "Gaadiidka FC",
        "Elman FC",
        "Heegan SC",
        "Jeenyo FC",
        "Mogadishu City Club",
        "Dekedda SC"
    ]

    # Define Seria A-B teams
    SERIA_AB_TEAMS = [
        "Raadsan FC",
        "Badbaado FC",
        "Batroolka FC",
        "Rajo FC",
        "Geeska Afrika FC",
        "Waxool FC",
        "Jazeera SC",
        "Midnimo FC",
        "SomaliFruit FC",
        "Sahafi FC"
    ]

    # Streamlit UI
    st.markdown('<h1 class="main-title">Somali Football Match Predictor</h1>', unsafe_allow_html=True)

    team_names = ["Select a team"] + df_team_strength.index.tolist()

    if "home_team" not in st.session_state:
        st.session_state.home_team = "Select a team"

    if "away_team" not in st.session_state:
        st.session_state.away_team = "Select a team"

    col1, col2 = st.columns(2)

    with col1:
        home_logo = get_team_logo(st.session_state.home_team) if st.session_state.home_team != "Select a team" else None
        if home_logo:
            st.image(create_circular_logo(home_logo), caption=st.session_state.home_team, use_container_width=False)

    with col2:
        away_logo = get_team_logo(st.session_state.away_team) if st.session_state.away_team != "Select a team" else None
        if away_logo:
            st.image(create_circular_logo(away_logo), caption=st.session_state.away_team, use_container_width=False)

    sel1, sel2 = st.columns(2)

    with sel1:
        home_team = st.selectbox("Select Home Team", options=team_names, index=team_names.index(st.session_state.home_team), key='home_team')
        # Show team classification below home select
        if home_team != "Select a team":
            seria_a_teams_lower = [t.strip().lower() for t in SERIA_A_TEAMS]
            seria_ab_teams_lower = [t.strip().lower() for t in SERIA_AB_TEAMS]
            if home_team.strip().lower() in seria_a_teams_lower:
                st.markdown('<div class="team-label seria-a">All time Seria A</div>', unsafe_allow_html=True)
            elif home_team.strip().lower() in seria_ab_teams_lower:
                st.markdown('<div class="team-label seria-ab">Seria A-B</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="team-label seria-bc">Seria B-C</div>', unsafe_allow_html=True)

    if home_team != "Select a team":
        away_team_options = ["Select a team"] + [team for team in df_team_strength.index if team != home_team]
        if st.session_state.away_team == home_team or st.session_state.away_team not in away_team_options:
            st.session_state.away_team = away_team_options[1] if len(away_team_options) > 1 else "Select a team"
    else:
        away_team_options = ["Select a team"] + df_team_strength.index.tolist()

    with sel2:
        away_team = st.selectbox("Select Away Team", options=away_team_options, index=away_team_options.index(st.session_state.away_team), key='away_team')
        # Show team classification below away select
        if away_team != "Select a team":
            seria_a_teams_lower = [t.strip().lower() for t in SERIA_A_TEAMS]
            seria_ab_teams_lower = [t.strip().lower() for t in SERIA_AB_TEAMS]
            if away_team.strip().lower() in seria_a_teams_lower:
                st.markdown('<div class="team-label seria-a">All time Seria A</div>', unsafe_allow_html=True)
            elif away_team.strip().lower() in seria_ab_teams_lower:
                st.markdown('<div class="team-label seria-ab">Seria A-B</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="team-label seria-bc">Seria B-C</div>', unsafe_allow_html=True)

    def clear_selection():
        st.session_state.home_team = "Select a team"
        st.session_state.away_team = "Select a team"

    # Center the buttons in a fixed bottom-right container
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_button = st.button("Predict Match Outcome")
    with col2:
        clear_button = st.button("Clear Selection", on_click=clear_selection, key="clear_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Modal logic for Seria A vs Seria B-C warning ---
    # (Remove all modal/session state logic)
    # --- End modal logic ---

    # Show prediction results
    if predict_button:
        # (Remove Seria A vs Seria B-C modal check)
        if home_team != "Select a team" and away_team != "Select a team" and validate_teams(home_team, away_team, df):
            
            # Calculate predictions
            h2h_info = get_h2h_info(home_team, away_team, df)
            
            # Calculate logical predictions based on data
            h2h_info = get_h2h_info(home_team, away_team, df)
            home_form = get_team_form(home_team, df)
            away_form = get_team_form(away_team, df)
            
            # Initialize base probabilities
            home_prob = 33.33
            away_prob = 33.33
            draw_prob = 33.34
            
            # Rule 1: Check Head-to-Head record first
            if h2h_info['total_matches'] > 0:
                print(f"H2H Record: {home_team} {h2h_info['home_wins']} wins, {away_team} {h2h_info['away_wins']} wins, {h2h_info['draws']} draws")
                
                total_h2h = h2h_info['total_matches']
                home_h2h_ratio = h2h_info['home_wins'] / total_h2h
                away_h2h_ratio = h2h_info['away_wins'] / total_h2h
                draw_h2h_ratio = h2h_info['draws'] / total_h2h
                
                # Apply H2H-based predictions with home advantage
                if home_h2h_ratio > away_h2h_ratio:
                    # Home team has better H2H record
                    home_prob = 50 + (home_h2h_ratio * 20)  # 50-70%
                    away_prob = 25 + (away_h2h_ratio * 15)  # 25-40%
                    draw_prob = 100 - home_prob - away_prob
                elif away_h2h_ratio > home_h2h_ratio:
                    # Away team has better H2H record
                    away_prob = 45 + (away_h2h_ratio * 20)  # 45-65%
                    home_prob = 30 + (home_h2h_ratio * 15)  # 30-45%
                    draw_prob = 100 - home_prob - away_prob
                else:
                    # Equal H2H record, favor draw slightly
                    draw_prob = 40 + (draw_h2h_ratio * 20)  # 40-60%
                    home_prob = (100 - draw_prob) * 0.6  # Split remaining
                    away_prob = (100 - draw_prob) * 0.4
                    
            # Rule 2: If no H2H matches, use recent form (last 5 matches)
            else:
                print(f"No H2H matches. Using recent form: {home_team} {home_form['points']} pts, {away_team} {away_form['points']} pts")
                
                home_points = home_form['points']
                away_points = away_form['points']
                total_points = home_points + away_points
                
                if total_points > 0:
                    home_form_ratio = home_points / total_points
                    away_form_ratio = away_points / total_points
                    
                    # Apply form-based predictions with home advantage
                    if home_form_ratio > away_form_ratio:
                        # Home team has better form
                        home_prob = 50 + (home_form_ratio * 25)  # 50-75%
                        away_prob = 20 + (away_form_ratio * 20)  # 20-40%
                        draw_prob = 100 - home_prob - away_prob
                    elif away_form_ratio > home_form_ratio:
                        # Away team has better form
                        away_prob = 45 + (away_form_ratio * 25)  # 45-70%
                        home_prob = 25 + (home_form_ratio * 20)  # 25-45%
                        draw_prob = 100 - home_prob - away_prob
                    else:
                        # Equal form, slight home advantage
                        home_prob = 40
                        away_prob = 35
                        draw_prob = 25
                else:
                    # No recent form data, use home advantage
                    home_prob = 45
                    away_prob = 35
                    draw_prob = 20
            
            # Ensure probabilities sum to 100%
            total = home_prob + draw_prob + away_prob
            home_prob = (home_prob / total) * 100
            draw_prob = (draw_prob / total) * 100
            away_prob = (away_prob / total) * 100
            
            print(f"Final predictions: {home_team} {home_prob:.1f}%, Draw {draw_prob:.1f}%, {away_team} {away_prob:.1f}%")

            # Determine the predicted outcome
            max_prob = max(home_prob, draw_prob, away_prob)
            if max_prob == home_prob:
                predicted_outcome = f"{home_team} wins"
            elif max_prob == away_prob:
                predicted_outcome = f"{away_team} wins"
            else:
                predicted_outcome = "Draw"

            # Determine dynamic colors for display
            lead_color = "#10b981"  # green for higher percentage team
            trail_color = "#ef4444"  # red for lower percentage team
            draw_color = "#f59e0b"   # yellow for draw

            if home_prob > away_prob:
                home_color = lead_color
                away_color = trail_color
            elif away_prob > home_prob:
                home_color = trail_color
                away_color = lead_color
            else:
                # equal probabilities for teams
                home_color = trail_color
                away_color = trail_color

            # Display match prediction with percentages
            st.markdown('<div class="section-title">Match Prediction</div>', unsafe_allow_html=True)
            
            # Display prediction percentages
            st.markdown('<div class="prediction-bar-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Probabilities</div>', unsafe_allow_html=True)
            
            # Create prediction bar
            total_width = 100
            home_width = home_prob
            draw_width = draw_prob
            away_width = away_prob
            
            st.markdown(f'''
            <div class="prediction-bar-wrapper">
                <div class="home-segment" style="width: {home_width}%; height: 100%; float: left; background: {home_color};"></div>
                <div class="draw-segment" style="width: {draw_width}%; height: 100%; float: left; background: {draw_color};"></div>
                <div class="away-segment" style="width: {away_width}%; height: 100%; float: left; background: {away_color};"></div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display percentages
            st.markdown(f'''
            <div class="prediction-percentages">
                <span class="percentage-label" style="color: {home_color}; border: 2px solid {home_color};">{home_team}: {home_prob:.1f}%</span>
                <span class="percentage-label" style="color: {draw_color}; border: 2px solid {draw_color};">Draw: {draw_prob:.1f}%</span>
                <span class="percentage-label" style="color: {away_color}; border: 2px solid {away_color};">{away_team}: {away_prob:.1f}%</span>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Head-to-Head Information
            st.markdown('<div class="section-title">Head-to-Head Statistics</div>', unsafe_allow_html=True)
            
            if h2h_info['total_matches'] > 0:
                h2h_col1, h2h_col2, h2h_col3, h2h_col4 = st.columns(4)
                
                with h2h_col1:
                    st.markdown(f'''
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['total_matches']}</div>
                        <div class="stat-label">Total Matches</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with h2h_col2:
                    st.markdown(f'''
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['home_wins']}</div>
                        <div class="stat-label">{home_team} Wins</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with h2h_col3:
                    st.markdown(f'''
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['away_wins']}</div>
                        <div class="stat-label">{away_team} Wins</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with h2h_col4:
                    st.markdown(f'''
                    <div class="stat-item">
                        <div class="stat-value">{h2h_info['draws']}</div>
                        <div class="stat-label">Draws</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center;color:#64748b;font-style:italic;">No previous head-to-head matches found</div>', unsafe_allow_html=True)
            
            # Display Last 5 Matches for each team
            st.markdown('<div class="section-title">Recent Form (Last 5 Matches)</div>', unsafe_allow_html=True)
            
            # Get last 5 matches for each team
            home_matches = get_last_5_matches_details(home_team, df)
            away_matches = get_last_5_matches_details(away_team, df)
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                st.markdown(f'<div class="team-name">{home_team}</div>', unsafe_allow_html=True)
                st.markdown('<div style="margin-top: 15px;"><strong>Last 5 Matches:</strong></div>', unsafe_allow_html=True)
                if home_matches:
                    for match in home_matches:
                        venue = "H" if match['is_home'] else "A"
                        st.markdown(f'''
                        <div class="match-result">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>{match['opponent']} ({venue})</span>
                                <span class="result-indicator {match['result_class']}">{match['result']}</span>
                            </div>
                            <div style="text-align: center; font-size: 14px; color: #64748b; margin-top: 5px;">
                                {match['team_score']} - {match['opponent_score']} ({match['season']})
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#64748b;font-style:italic;">No recent matches found.</div>', unsafe_allow_html=True)
            
            with form_col2:
                st.markdown(f'<div class="team-name">{away_team}</div>', unsafe_allow_html=True)
                st.markdown('<div style="margin-top: 15px;"><strong>Last 5 Matches:</strong></div>', unsafe_allow_html=True)
                if away_matches:
                    for match in away_matches:
                        venue = "H" if match['is_home'] else "A"
                        st.markdown(f'''
                        <div class="match-result">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>{match['opponent']} ({venue})</span>
                                <span class="result-indicator {match['result_class']}">{match['result']}</span>
                            </div>
                            <div style="text-align: center; font-size: 14px; color: #64748b; margin-top: 5px;">
                                {match['team_score']} - {match['opponent_score']} ({match['season']})
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#64748b;font-style:italic;">No recent matches found.</div>', unsafe_allow_html=True)

            # Add some spacing between sections
            st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

        # Example: inside your prediction button callback
        



            

        save_prediction_to_history(home_team, away_team, home_prob, draw_prob, away_prob, predicted_outcome)



# End of prediction results
            



            

            
elif page == "📜 History":

    
    
    # --- Load history from MongoDB ---
    history = load_prediction_history()   # already defined earlier


    # Convert image file to base64 data URI for embedding in HTML
    def img_to_data_uri(img_path):
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        return None

    def get_team_logo(team_name):
        logo_folder = "TeamLogo"
        for ext in ["png", "jpeg", "jpg"]:
            logo_path = os.path.join(logo_folder, f"{team_name}.{ext}")
            if os.path.exists(logo_path):
                return logo_path
        return None

    def _safe_rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()


    # --- handle delete requests ---
    if "delete_id" in st.session_state:
        delete_id = st.session_state.pop("delete_id")
        collection.delete_one({"_id": delete_id})   # delete from MongoDB
        _safe_rerun()


    # --- show prediction history ---
    if history:
        st.markdown("### Prediction History")

        for idx, entry in enumerate(reversed(history)):
            home_team = entry["home_team"]
            away_team = entry["away_team"]
            home_prob = entry["home_prob"]
            draw_prob = entry["draw_prob"]
            away_prob = entry["away_prob"]
            predicted_outcome = entry["predicted_outcome"]

            lead_color = "#10b981"
            trail_color = "#ef4444"
            draw_color = "#f59e0b"

            if home_prob > away_prob:
                home_color = lead_color
                away_color = trail_color
            elif away_prob > home_prob:
                home_color = trail_color
                away_color = lead_color
            else:
                home_color = trail_color
                away_color = trail_color

            home_logo = img_to_data_uri(get_team_logo(home_team))
            away_logo = img_to_data_uri(get_team_logo(away_team))

            # Convert ObjectId to string for button key
            mongo_id = entry["_id"]

            # Card top part (everything except delete button)
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 0;">
                        <div style="text-align:center; flex:1;">
                            <img src="{home_logo}" style="width:60px; height:60px; border-radius:50%;"><br>
                            <b>{home_team}</b>
                        </div>
                        <div style="flex:0.2; text-align:center; font-weight:bold;">vs</div>
                        <div style="text-align:center; flex:1;">
                            <img src="{away_logo}" style="width:60px; height:60px; border-radius:50%;"><br>
                            <b>{away_team}</b>
                        </div>
                    </div>
                    <div style="margin-top:8px; text-align:center;  font-size:13px; color:#64748b;">{entry['timestamp']}</div>
                    <div style="margin-top:10px; text-align:center; ">
                        <span style="color:{home_color};   border:2px solid {home_color}; border-radius:6px; padding:4px 10px; margin:5px;">
                            {home_team}: {home_prob:.1f}%
                        </span>
                        <span style="color:{draw_color};  border:2px solid {draw_color}; border-radius:6px; padding:4px 10px; margin:5px;">
                            Draw: {draw_prob:.1f}%
                        </span>
                        <span style="color:{away_color};   border:2px solid {away_color}; border-radius:6px; padding:4px 10px; margin:5px;">
                            {away_team}: {away_prob:.1f}%
                        </span>
                    </div>
                    <div style="margin-top:10px; text-align:center;  font-style:italic; color:#64748b;">
                        Predicted Outcome: <b>{predicted_outcome}</b>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Delete button
            btn_key = f"delete_{mongo_id}"
            if st.button("🗑 Delete", key=btn_key):
                from bson import ObjectId
                st.session_state["delete_id"] = ObjectId(mongo_id)
                _safe_rerun()

            st.markdown(
                """
                <style>
                div[data-testid="stButton"] > button[kind="secondary"] {
                    background-color: #ef4444 !important;
                    color: white !important;
                    border: none;
                    border-radius: 8px;
                    width: 100%;
                    padding: 10px 12px;
                    font-weight: 500;
                    cursor: pointer;
                    margin-top: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Close the card container
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("No predictions in history yet.")








# Add this at the beginning of your file, after the initial imports
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Add theme toggle button in the sidebar
with st.sidebar:
    if st.button('🌓 Toggle Theme'):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Add CSS for theme colors
st.markdown(f"""
    <style>
    /* Theme colors */
    :root {{
        --background-color: {('#0e1117' if st.session_state.theme == 'dark' else '#ffffff')};
        --text-color: {('#fafafa' if st.session_state.theme == 'dark' else '#262730')};
        --secondary-background: {('#262730' if st.session_state.theme == 'dark' else '#f0f2f6')};
    }}

    /* Apply theme to Streamlit elements */
    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}

    .centered-container {{
        background-color: var(--secondary-background);
    }}

    .prediction-results {{
        background-color: var(--secondary-background);
    }}

    .match-result {{
        background-color: var(--background-color);
    }}

    .stat-item {{
        background-color: var(--background-color);
    }}

    /* Update text colors based on theme */
    .team-name, .section-title, .stat-value {{
        color: var(--text-color);
    }}
    </style>
""", unsafe_allow_html=True)

# Add enhanced CSS styling
st.markdown("""
    <style>
    /* Main container styling */
    .centered-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: var(--text-color) !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 30px;
        text-shadow: none;
        background: none;
        -webkit-background-clip: unset;
        -webkit-text-fill-color: unset;
        background-clip: unset;
    }
    
    /* Team selection area */
    .team-selection-area {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 20px 0;
    }
    
    .logo-item {
        text-align: center;
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .logo-item:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction results styling */
    .prediction-results {
        background: rgba(255,255,255,0.95);
        padding: 30px;
        border-radius: 20px;
        margin-top: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .section-title {
        color: var(--text-color);
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin: 10px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        position: relative;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Prediction bar styling */
    .prediction-bar-container {
        background: white;
        border: none;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .prediction-bar-wrapper {
        height: 16px;
        background: #f1f5f9;
        border-radius: 10px;
        overflow: hidden;
        margin: 15px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .home-segment {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
    }
    
    .draw-segment {
        background: linear-gradient(90deg, #7c3aed, #6d28d9);
    }
    
    .away-segment {
        background: linear-gradient(90deg, #16a34a, #15803d);
    }
    
    /* Match result cards */
    .match-result {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .match-result:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .result-indicator {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-win {
        background: linear-gradient(45deg, #10b981, #059669);
        color: white;
        box-shadow: 0 2px 8px rgba(16,185,129,0.3);
    }
    
    .result-loss {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 2px 8px rgba(239,68,68,0.3);
    }
    
    .result-draw {
        background: linear-gradient(45deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 2px 8px rgba(245,158,11,0.3);
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 25px 0;
        text-align: center;
        flex-wrap: wrap;
    }
    
    .stat-item {
        padding: 20px 30px;
        background: white;
        border-radius: 15px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        min-width: 120px;
    }
    
    .stat-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .stat-value {
        font-size: 32px;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 8px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .team-name {
        font-weight: 700;
        color: #2c3e50;
        font-size: 18px;
        text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Percentage styling */
    .prediction-percentages {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        font-weight: 700;
        font-size: 16px;
    }
    
    .percentage-label {
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 25px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .percentage-label:hover {
        transform: translateY(-2px);
    }
    
    .home-percentage {
        color: #2563eb;
        border: 2px solid #2563eb;
    }
    
    .draw-percentage {
        color: #7c3aed;
        border: 2px solid #7c3aed;
    }
    
    .away-percentage {
        color: #16a34a;
        border: 2px solid #16a34a;
    }
    
    /* Team labels */
    .team-label {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 700;
        margin: 10px auto 15px auto;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: transform 0.3s ease;
    }
    
    .team-label:hover {
        transform: translateY(-2px);
    }
    
    .team-label.seria-a {
        background: linear-gradient(45deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(37,99,235,0.3);
    }
    
    .team-label.seria-ab {
        background: linear-gradient(45deg, #7c3aed, #6d28d9);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3);
    }
    
    .team-label.seria-bc {
        background: linear-gradient(45deg, #ef6c00, #ea580c);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(239,108,0,0.3);
    }
    
    /* Button styling */
    .button-container {
        position: fixed;
        bottom: 40px;
        right: 40px;
        z-index: 1000;
        display: flex;
        gap: 20px;
        justify-content: flex-end;
        background: transparent;
    }
    
    .stButton > button {
        color: white !important;
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border-radius: 25px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: none !important;
        padding: 12px 30px !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a67d8, #6b46c1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(102,126,234,0.4) !important;
    }
    
    .stButton.clear-btn > button {
        background: linear-gradient(45deg, #ef6c00, #ea580c) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 25px rgba(239,108,0,0.3) !important;
    }
    
    .stButton.clear-btn > button:hover {
        background: linear-gradient(45deg, #ea580c, #dc2626) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(239,108,0,0.4) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.1) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .centered-container {
            padding: 20px;
            margin: 10px;
        }
        
        .logo-container {
            flex-direction: column;
            gap: 20px;
        }
        
        .stats-container {
            gap: 15px;
        }
        
        .prediction-percentages {
            flex-direction: column;
            gap: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

