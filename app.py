# =============================================================================
# app.py — Flask REST API Backend
# =============================================================================
# PURPOSE:
#   This is the backend server for the Movie Success Predictor app.
#   It loads your trained Gradient Boosting model and exposes a single
#   endpoint: POST /predict
#   The Streamlit frontend calls this endpoint and displays the result.
#
# HOW TO RUN (in your VS Code terminal, from the Capstone folder):
#   python3 app.py
#
# The server will start at: http://127.0.0.1:5000
# Keep this terminal running while you use the Streamlit app.
# =============================================================================


# --- Import Libraries ---------------------------------------------------------
from flask import Flask, request, jsonify   # Flask: web framework for the API
import joblib                               # joblib: loads the saved .pkl files
import numpy as np                          # numpy: for numerical operations
import pandas as pd                         # pandas: for building the input DataFrame
import os                                   # os: for building file paths
import traceback                            # traceback: for detailed error messages

# --- Initialise Flask App -----------------------------------------------------
# Flask(__name__) creates a new web application.
# __name__ tells Flask where to find resources (templates, static files, etc.)

app = Flask(__name__)

# =============================================================================
# SECTION 1: LOAD MODEL FILES AT STARTUP
# =============================================================================
# We load the model, scaler, and feature names ONCE when the server starts.
# This is more efficient than reloading them on every request.
# All three files must be in the same folder as this app.py script.

# Build the path to the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths for the three .pkl files exported from your notebook
MODEL_PATH         = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH        = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

print("=" * 55)
print("  🎬 Movie Success Predictor — Flask API")
print("  Group 6 | PGP DSE GenAI Sept 2025")
print("=" * 55)

# Load the trained Gradient Boosting model
print("\n[1/3] Loading model.pkl ...")
model = joblib.load(MODEL_PATH)
print(f"      ✅ Model loaded — {model.n_estimators_} estimators, "
      f"{model.n_features_in_} features")

# Load the fitted StandardScaler (saved from your notebook)
# NOTE: Gradient Boosting was trained on UNSCALED data in your notebook.
# The scaler is loaded here for reference / future use but we do NOT
# apply it to the input before passing to the GBM model.
print("[2/3] Loading scaler.pkl ...")
scaler = joblib.load(SCALER_PATH)
print(f"      ✅ Scaler loaded")

# Load the exact feature column names in the correct training order
# This is critical — the model expects columns in exactly this order.
print("[3/3] Loading feature_names.pkl ...")
FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
print(f"      ✅ {len(FEATURE_NAMES)} feature names loaded")
print(f"\n      Features: {FEATURE_NAMES}\n")

# =============================================================================
# SECTION 2: FEATURE IMPORTANCE — precompute at startup
# =============================================================================
# Extract the feature importances from the trained model.
# We use these to return the top 5 drivers of each prediction.
# pd.Series lets us sort and label them easily.

feature_importance_series = pd.Series(
    model.feature_importances_,
    index=FEATURE_NAMES
).sort_values(ascending=False)

print("=" * 55)
print("  Top 10 features by importance:")
for feat, imp in feature_importance_series.head(10).items():
    print(f"    {feat:<35} {imp:.4f}")
print("=" * 55)


# =============================================================================
# SECTION 3: HELPER — BUILD FEATURE VECTOR FROM RAW USER INPUT
# =============================================================================
# This function takes the raw values from the Streamlit form (budget in USD,
# genre list, language, etc.) and converts them into the exact same format
# that was used during model training in your notebook.
#
# It mirrors the feature engineering steps from Section 5 of your notebook.

def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Convert raw user input (from the Streamlit form / API request body)
    into a single-row DataFrame matching the model's training feature matrix.

    Parameters
    ----------
    data : dict
        Raw input values from the frontend. Keys are described below.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns in exactly the same order as FEATURE_NAMES.
    """

    # --- Extract raw inputs from the request body ----------------------------
    # These are the values the user typed / selected in the Streamlit sidebar.

    budget_usd     = float(data.get("budget_usd", 10_000_000))   # Raw USD budget
    popularity     = float(data.get("popularity", 20.0))          # TMDB popularity score
    avg_vote       = float(data.get("avg_vote", 6.5))             # IMDb avg rating (0-10)
    vote_count     = float(data.get("vote_count", 1000))          # Number of IMDb votes
    num_votes      = float(data.get("num_votes", 1000))           # External rating vote count
    runtime        = float(data.get("runtime", 105))              # Film duration in minutes
    release_month  = int(data.get("release_month", 6))            # 1-12
    release_year   = int(data.get("release_year", 2023))          # e.g. 2023
    release_day    = int(data.get("release_day", 15))             # 1-31
    is_english     = int(data.get("is_english", 1))               # 1=English, 0=Other
    has_tagline    = int(data.get("has_tagline", 1))              # 1=has tagline, 0=no
    is_adult       = int(data.get("is_adult", 0))                 # 1=adult content, 0=no
    num_cast       = int(data.get("num_cast", 8))                 # Number of cast members
    num_directors  = int(data.get("num_directors", 1))            # Number of directors
    num_writers    = int(data.get("num_writers", 2))              # Number of writers
    num_prod_cos   = int(data.get("num_production_companies", 2)) # Number of production companies
    num_spoken_langs = int(data.get("num_spoken_languages", 1))   # Number of spoken languages
    lang_freq      = float(data.get("lang_freq", 3000))           # Language frequency in dataset
    genres_selected = data.get("genres", [])                      # List of selected genre strings

    # --- Apply log1p transformations ------------------------------------------
    # These mirror the exact transformations in Section 5 of your notebook:
    #   df['Log_Budget']     = np.log1p(df['Budget'])
    #   df['Log_Popularity'] = np.log1p(df['Popularity'])
    #   df['Log_Vote_Count'] = np.log1p(df['Vote_Count'])
    #   df['Log_Num_Votes']  = np.log1p(df['Num_Votes'])
    # np.log1p(x) = log(1 + x) — handles zero values safely

    log_budget      = np.log1p(budget_usd)
    log_popularity  = np.log1p(popularity)
    log_vote_count  = np.log1p(vote_count)
    log_num_votes   = np.log1p(num_votes)

    # --- Derive temporal features ---------------------------------------------
    # These mirror Section 5 of your notebook:
    #   df['Is_Summer']    = df['Release_Month'].isin([6, 7, 8]).astype(int)
    #   df['Is_Holiday']   = df['Release_Month'].isin([11, 12]).astype(int)
    #   df['Is_WeekStart'] = df['Release_Date'].dt.dayofweek.isin([0, 1]).astype(int)
    #   df['Movie_Age']    = 2024 - df['Release_Year']

    is_summer    = 1 if release_month in [6, 7, 8]   else 0
    is_holiday   = 1 if release_month in [11, 12]    else 0
    is_weekstart = 0   # Default — not derivable from month alone; set to 0
    movie_age    = 2024 - release_year

    # --- Vote Quality (binned Avg_Vote) ---------------------------------------
    # Mirrors: pd.cut(df['Avg_Vote'], bins=[0,5,6,7,8,10], labels=[0,1,2,3,4])
    # Converts the continuous avg_vote score into an ordered quality tier.

    if avg_vote <= 5:
        vote_quality = 0.0
    elif avg_vote <= 6:
        vote_quality = 1.0
    elif avg_vote <= 7:
        vote_quality = 2.0
    elif avg_vote <= 8:
        vote_quality = 3.0
    else:
        vote_quality = 4.0

    # --- Genre Binary Flags ---------------------------------------------------
    # Mirrors Section 5 of your notebook:
    #   TOP_GENRES = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure',
    #                 'Science Fiction', 'Romance', 'Horror', 'Animation', 'Crime']
    # For each genre, 1 if selected by the user, else 0.
    # Note: 'Science Fiction' maps to column name 'Genre_Science_Fiction'
    #       (the notebook replaces the space with underscore)

    TOP_GENRES = [
        'Action', 'Drama', 'Comedy', 'Thriller', 'Adventure',
        'Science Fiction', 'Romance', 'Horror', 'Animation', 'Crime'
    ]

    genre_flags = {}
    for genre in TOP_GENRES:
        col_name = f"Genre_{genre.replace(' ', '_')}"   # e.g. "Genre_Science_Fiction"
        genre_flags[col_name] = 1 if genre in genres_selected else 0

    # --- Genre Count ----------------------------------------------------------
    # Total number of genres selected by the user.
    # Mirrors: df['Genre_Count'] = df['Genres'].apply(lambda x: len(...))

    genre_count = len(genres_selected)

    # --- Avg_Rating -----------------------------------------------------------
    # In your notebook, Avg_Rating is a separate column from the raw dataset.
    # It is highly correlated with Avg_Vote (r ≈ 0.97).
    # We use Avg_Vote as a proxy for Avg_Rating since they're nearly identical.

    avg_rating = avg_vote

    # --- Build the full feature dictionary ------------------------------------
    # Every key here must exactly match a column name in FEATURE_NAMES.
    # The order doesn't matter here — we'll reorder using FEATURE_NAMES below.

    feature_dict = {
        # Raw numerical features (as they appear in X after feature engineering)
        "Avg_Vote":                  avg_vote,
        "Vote_Count":                vote_count,
        "Runtime":                   runtime,
        "Avg_Rating":                avg_rating,
        "Num_Votes":                 num_votes,
        "Popularity":                popularity,

        # Log-transformed features
        "Log_Budget":                log_budget,
        "Log_Popularity":            log_popularity,
        "Log_Vote_Count":            log_vote_count,
        "Log_Num_Votes":             log_num_votes,

        # Date-derived features
        "Release_Year":              release_year,
        "Release_Month":             release_month,
        "Release_Day":               release_day,
        "Is_Summer":                 is_summer,
        "Is_Holiday":                is_holiday,
        "Is_WeekStart":              is_weekstart,
        "Movie_Age":                 movie_age,

        # Language features
        "Is_English":                is_english,
        "Lang_Freq":                 lang_freq,

        # Binary flags
        "Has_Tagline":               has_tagline,
        "Is_Adult":                  is_adult,

        # Crew & production counts
        "Num_Cast":                  num_cast,
        "Num_Directors":             num_directors,
        "Num_Writers":               num_writers,
        "Num_Production_Companies":  num_prod_cos,
        "Num_Spoken_Languages":      num_spoken_langs,

        # Vote quality tier
        "Vote_Quality":              vote_quality,

        # Genre count
        "Genre_Count":               genre_count,

        # Genre binary flags (10 genres from your notebook's TOP_GENRES)
        **genre_flags,
    }

    # --- Build a single-row DataFrame -----------------------------------------
    # We create the DataFrame using only the columns in FEATURE_NAMES,
    # in exactly the same order as during training.
    # Any column in FEATURE_NAMES not in feature_dict defaults to 0.

    row = {}
    for col in FEATURE_NAMES:
        row[col] = feature_dict.get(col, 0)   # Default to 0 if column not found

    df_input = pd.DataFrame([row])

    return df_input


# =============================================================================
# SECTION 4: /predict ENDPOINT
# =============================================================================
# This is the main API endpoint. The Streamlit app sends a POST request here
# with a JSON body containing all the movie input parameters.
# We preprocess the inputs, run the model, and return the prediction.

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON input, returns prediction + confidence + top 5 features.
    """

    try:
        # --- Step 1: Parse the incoming JSON request body --------------------
        # request.get_json() reads the JSON body sent by the Streamlit frontend.
        # If the body is missing or malformed, we return a 400 error.

        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON body received. "
                            "Send Content-Type: application/json"}), 400

        print(f"\n📥 Received request: {data}")

        # --- Step 2: Input validation -----------------------------------------
        # Check that the budget is a positive number.
        # More validations can be added here as needed.

        budget_usd = float(data.get("budget_usd", 0))
        if budget_usd <= 0:
            return jsonify({"error": "budget_usd must be a positive number"}), 400

        popularity = float(data.get("popularity", 0))
        if popularity < 0:
            return jsonify({"error": "popularity must be 0 or greater"}), 400

        avg_vote = float(data.get("avg_vote", 6.5))
        if not (0 <= avg_vote <= 10):
            return jsonify({"error": "avg_vote must be between 0 and 10"}), 400

        release_month = int(data.get("release_month", 6))
        if not (1 <= release_month <= 12):
            return jsonify({"error": "release_month must be between 1 and 12"}), 400

        # --- Step 3: Build the feature vector --------------------------------
        # Call our helper function to convert raw inputs into the model's
        # expected feature format (same engineering as your notebook).

        df_input = build_feature_vector(data)
        print(f"📊 Feature vector shape: {df_input.shape}")
        print(f"📊 Feature vector:\n{df_input.T}")

        # --- Step 4: Run the model -------------------------------------------
        # model.predict()       returns 0 (Unsuccessful) or 1 (Successful)
        # model.predict_proba() returns [prob_unsuccessful, prob_successful]
        # NOTE: We pass df_input directly (unscaled) because your Gradient
        # Boosting model was trained on unscaled data (X_train.values).

        prediction    = model.predict(df_input)[0]           # 0 or 1
        probabilities = model.predict_proba(df_input)[0]     # [prob_0, prob_1]
        confidence    = float(probabilities[1])               # Probability of success

        # Convert prediction to human-readable label
        label = "Successful" if prediction == 1 else "Unsuccessful"

        print(f"🎯 Prediction: {label} | Confidence: {confidence:.2%}")

        # --- Step 5: Get top 5 feature importances ---------------------------
        # We return the top 5 most important features globally (from the trained
        # model) so the Streamlit app can display them as a bar chart.
        # These are the same for every prediction (global importances).

        top5 = feature_importance_series.head(5)
        top5_list = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in top5.items()
        ]

        # --- Step 6: Build and return the JSON response ----------------------
        # The Streamlit app reads these exact keys from the response.

        response = {
            "prediction":   label,                        # "Successful" or "Unsuccessful"
            "confidence":   round(confidence * 100, 1),   # e.g. 73.4 (as percentage)
            "label":        "✅ GREENLIGHT" if prediction == 1 else "❌ PASS",
            "top_features": top5_list,                    # List of {feature, importance}
            "input_echo":   {                             # Echo back key inputs for display
                "budget_usd":     budget_usd,
                "popularity":     float(data.get("popularity", 20)),
                "avg_vote":       float(data.get("avg_vote", 6.5)),
                "release_month":  release_month,
                "genres":         data.get("genres", []),
            }
        }

        print(f"📤 Response: {response}")
        return jsonify(response), 200

    except Exception as e:
        # If anything goes wrong, return a 500 error with the full traceback
        # so you can debug it easily.
        error_msg = traceback.format_exc()
        print(f"❌ ERROR:\n{error_msg}")
        return jsonify({"error": str(e), "traceback": error_msg}), 500


# =============================================================================
# SECTION 5: HEALTH CHECK ENDPOINT
# =============================================================================
# A simple GET endpoint to confirm the server is running.
# Visit http://127.0.0.1:5000/health in your browser to check.

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "running",
        "model":     "Gradient Boosting Classifier",
        "features":  len(FEATURE_NAMES),
        "test_auc":  0.874,
        "test_f1":   0.861,
        "group":     "Group 6 | PGP DSE GenAI Sept 2025"
    }), 200


# =============================================================================
# SECTION 6: RUN THE SERVER
# =============================================================================
# debug=True  : shows detailed error messages in the terminal (turn off in production)
# port=5000   : the Streamlit app will call http://127.0.0.1:5000/predict
# host='0.0.0.0' is not needed for local use — 127.0.0.1 (localhost) is fine.

if __name__ == "__main__":
    print("\n🚀 Starting Flask server on http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000)