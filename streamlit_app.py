# =============================================================================
# streamlit_app.py — Cinecast Movie Success Predictor (Redesigned)
# =============================================================================
# CHANGES FROM v1:
#   - Removed model metric badges (AUC, F1, Accuracy, Benchmarked)
#   - Removed sidebar; all inputs are now in a 3-step scrollable main page
#   - "PASS" verdict renamed to "FAIL"
#   - Results section replaced with professional executive-grade output:
#       • Short verdict card (hybrid: bold summary + expandable detail)
#       • Actionable investment intelligence (not raw ML stats)
#   - AI chatbox added below results, context-aware of the prediction
#       • Suggested starter questions pre-seeded after prediction
#       • Free-text input also available
#
# HOW TO RUN:
#   1. Start Flask backend: python3 app.py
#   2. In another terminal: streamlit run streamlit_app.py
# =============================================================================

import streamlit as st
import requests
import json
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="🎬 Cinecast | Movie Investment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONSTANTS
# =============================================================================
API_URL = "http://127.0.0.1:5000/predict"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

FEATURE_DISPLAY_NAMES = {
    "Log_Num_Votes":             "Audience Engagement (Vote Count)",
    "Log_Budget":                "Production Budget",
    "Movie_Age":                 "Film Age / Legacy",
    "Num_Votes":                 "Raw Vote Count",
    "Release_Year":              "Release Year",
    "Vote_Count":                "IMDb Vote Count",
    "Log_Vote_Count":            "Log Vote Count",
    "Runtime":                   "Film Runtime",
    "Popularity":                "TMDB Popularity Score",
    "Avg_Vote":                  "Average IMDb Rating",
    "Log_Popularity":            "Log Popularity",
    "Genre_Action":              "Action Genre",
    "Genre_Adventure":           "Adventure Genre",
    "Genre_Animation":           "Animation Genre",
    "Genre_Comedy":              "Comedy Genre",
    "Genre_Drama":               "Drama Genre",
    "Genre_Thriller":            "Thriller Genre",
    "Genre_Horror":              "Horror Genre",
    "Genre_Science_Fiction":     "Science Fiction Genre",
    "Genre_Crime":               "Crime Genre",
    "Genre_Romance":             "Romance Genre",
    "Is_English":                "English Language",
    "Is_Summer":                 "Summer Release",
    "Is_Holiday":                "Holiday Release",
    "Num_Cast":                  "Cast Size",
    "Num_Directors":             "Number of Directors",
    "Num_Writers":               "Number of Writers",
    "Num_Production_Companies":  "Production Companies",
    "Lang_Freq":                 "Language Popularity",
    "Genre_Count":               "Number of Genres",
    "Has_Tagline":               "Has Marketing Tagline",
    "Avg_Rating":                "External Aggregate Rating",
    "Vote_Quality":              "Vote Quality Tier",
}

EXAMPLES = {
    "🦁 Blockbuster": {
        "desc": "High-budget action/adventure (think Avengers-style)",
        "budget_millions": 250.0, "popularity": 120.0, "avg_vote": 7.8,
        "vote_count": 500000, "runtime": 150, "release_month": 5,
        "release_year": 2023, "genres": ["Action", "Adventure", "Science Fiction"],
        "language": "English", "num_cast": 15, "num_directors": 1,
        "num_writers": 3, "num_prod_cos": 5, "num_spoken_langs": 1,
        "has_tagline": 1, "is_adult": 0
    },
    "🎭 Indie Drama": {
        "desc": "Low-budget drama with critical acclaim",
        "budget_millions": 2.0, "popularity": 8.0, "avg_vote": 7.2,
        "vote_count": 12000, "runtime": 95, "release_month": 10,
        "release_year": 2023, "genres": ["Drama", "Romance"],
        "language": "English", "num_cast": 5, "num_directors": 1,
        "num_writers": 1, "num_prod_cos": 1, "num_spoken_langs": 1,
        "has_tagline": 0, "is_adult": 0
    },
    "👻 Horror Hit": {
        "desc": "Mid-budget horror — high ROI genre",
        "budget_millions": 15.0, "popularity": 45.0, "avg_vote": 6.8,
        "vote_count": 80000, "runtime": 98, "release_month": 10,
        "release_year": 2023, "genres": ["Horror", "Thriller"],
        "language": "English", "num_cast": 8, "num_directors": 1,
        "num_writers": 2, "num_prod_cos": 2, "num_spoken_langs": 1,
        "has_tagline": 1, "is_adult": 0
    },
    "🌍 Foreign Film": {
        "desc": "Non-English arthouse production",
        "budget_millions": 5.0, "popularity": 12.0, "avg_vote": 7.5,
        "vote_count": 25000, "runtime": 120, "release_month": 3,
        "release_year": 2022, "genres": ["Drama", "Crime"],
        "language": "Other", "num_cast": 6, "num_directors": 1,
        "num_writers": 1, "num_prod_cos": 2, "num_spoken_langs": 2,
        "has_tagline": 0, "is_adult": 0
    },
}

# =============================================================================
# CUSTOM CSS — DARK CINEMATIC THEME
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #12080f 50%, #0a0a0f 100%);
        color: #e8e0d0;
    }

    /* ── Full-width layout ── */
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1rem !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        max-width: 100% !important;
    }

    /* Hide sidebar toggle and default nav */
    section[data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Main Title ── */
    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #d4af37, #f5d76e, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.0rem;
        color: #9a8a7a;
        text-align: center;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .divider-gold {
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4af37, #8b1a1a, #d4af37, transparent);
        margin: 1rem 0 2rem 0;
        border: none;
    }

    /* ── Step Headers ── */
    .step-header {
        font-family: 'Cinzel', serif;
        font-size: 1.2rem;
        color: #d4af37;
        background: linear-gradient(135deg, #1a0a05, #2a1005);
        border: 1px solid #3a2010;
        border-left: 4px solid #d4af37;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 2rem 0 1.2rem 0;
        letter-spacing: 0.05em;
    }

    .step-number {
        display: inline-block;
        background: #8b1a1a;
        color: #f5d76e;
        font-family: 'Cinzel', serif;
        font-size: 0.85rem;
        font-weight: 700;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        margin-right: 0.7rem;
    }

    /* ── Example Cards ── */
    .example-card {
        background: linear-gradient(135deg, #1a0f05, #0f0508);
        border: 1px solid #3a2a1a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: border-color 0.2s;
    }
    .example-card:hover { border-color: #d4af37; }
    .example-title { font-family: 'Cinzel', serif; font-size: 1.0rem; color: #d4af37; }
    .example-meta { font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #9a8a7a; margin-top: 0.3rem; }

    /* ── Verdict Cards ── */
    .verdict-greenlight {
        background: linear-gradient(135deg, #0d2b0d, #1a4a1a);
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(46, 204, 113, 0.3);
        margin: 1rem 0;
    }
    .verdict-fail {
        background: linear-gradient(135deg, #2b0d0d, #4a1a1a);
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(231, 76, 60, 0.3);
        margin: 1rem 0;
    }
    .verdict-text-green {
        font-family: 'Cinzel', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #2ecc71;
        letter-spacing: 0.1em;
        text-shadow: 0 0 20px rgba(46, 204, 113, 0.5);
    }
    .verdict-text-red {
        font-family: 'Cinzel', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #e74c3c;
        letter-spacing: 0.1em;
        text-shadow: 0 0 20px rgba(231, 76, 60, 0.5);
    }
    .confidence-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #d4af37;
        margin-top: 0.8rem;
        letter-spacing: 0.05em;
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Cinzel', serif;
        font-size: 1.2rem;
        color: #d4af37;
        border-bottom: 1px solid #3a2a0a;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        letter-spacing: 0.05em;
    }

    /* ── Intelligence Boxes ── */
    .intel-box {
        background: linear-gradient(135deg, #0f0f1a, #1a1020);
        border: 1px solid #2a2040;
        border-left: 4px solid #d4af37;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #c8c0b0;
        line-height: 1.75;
    }
    .intel-box h4 {
        font-family: 'Cinzel', serif;
        color: #d4af37;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        margin: 0 0 0.6rem 0;
        text-transform: uppercase;
    }
    .intel-tag {
        display: inline-block;
        background: #1a0a05;
        border: 1px solid #3a2010;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        color: #d4af37;
        margin: 0.2rem 0.2rem 0 0;
    }
    .risk-flag {
        background: linear-gradient(135deg, #2b1005, #3a1a05);
        border: 1px solid #8b3a1a;
        border-left: 4px solid #f39c12;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        margin: 0.8rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #e8c070;
    }
    .action-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1a1010;
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: #c8c0b0;
        line-height: 1.5;
    }
    .action-bullet {
        color: #d4af37;
        font-size: 1.1rem;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a0f05, #2a1a0a);
        border: 1px solid #3a2a0a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-value {
        font-family: 'Cinzel', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #d4af37;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #9a8a7a;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #8b1a1a, #c0392b);
        color: #f5d76e;
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        font-weight: 600;
        border: 1px solid #d4af37;
        border-radius: 8px;
        padding: 0.9rem 2rem;
        width: 100%;
        letter-spacing: 0.12em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #c0392b, #8b1a1a);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.4);
        transform: translateY(-1px);
    }

    /* ── Chat Box ── */
    .chat-section {
        background: linear-gradient(135deg, #0f0810, #1a0f1a);
        border: 1px solid #3a2040;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    .chat-header {
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        color: #d4af37;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #2a1a0a, #3a2010);
        border: 1px solid #5a3010;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0 0.5rem 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #e8e0d0;
    }
    .chat-bubble-ai {
        background: linear-gradient(135deg, #0f1520, #0a1020);
        border: 1px solid #2a3050;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 2rem 0.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #c8d0e0;
        line-height: 1.6;
    }
    .chat-label-user {
        font-size: 0.7rem;
        color: #d4af37;
        text-align: right;
        margin-bottom: 0.2rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.05em;
    }
    .chat-label-ai {
        font-size: 0.7rem;
        color: #6a7a9a;
        margin-bottom: 0.2rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.05em;
    }
    .suggestion-btn {
        background: #1a1030;
        border: 1px solid #3a2050;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #c0b0d0;
        cursor: pointer;
        margin: 0.3rem 0.3rem 0 0;
        transition: all 0.2s;
        display: inline-block;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #5a5a5a;
        text-align: center;
    }

    /* ── Slider & input label colors ── */
    .stSlider label, .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stRadio label {
        color: #c8b090 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }

    /* ── Mobile Responsive ── */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .main-title { font-size: 2rem !important; }
        .sub-title { font-size: 0.75rem !important; letter-spacing: 0.08em; }
        .step-header { font-size: 1rem !important; padding: 0.8rem 1rem !important; }
        .verdict-text-green, .verdict-text-red { font-size: 2.2rem !important; }
        .verdict-greenlight, .verdict-fail { padding: 1.5rem !important; }
        .intel-box { font-size: 0.85rem !important; padding: 1rem !important; }
        .metric-card { padding: 0.8rem 0.6rem !important; }
        .metric-value { font-size: 1.1rem !important; }
        .chat-bubble-user, .chat-bubble-ai {
            margin-left: 0.3rem !important;
            margin-right: 0.3rem !important;
            font-size: 0.85rem !important;
        }
        .example-card { padding: 0.7rem 0.8rem !important; }
        .example-title { font-size: 0.85rem !important; }
        .section-header { font-size: 1rem !important; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > div { min-width: 100% !important; width: 100% !important; }
        .stTextInput input { font-size: 16px !important; }
        .stButton > button { font-size: 0.85rem !important; padding: 0.7rem 0.8rem !important; }
        .disclaimer { font-size: 0.70rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "chat_input_val" not in st.session_state:
    st.session_state.chat_input_val = ""
if "current_step" not in st.session_state:
    st.session_state.current_step = 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def call_api(params: dict) -> dict:
    try:
        response = requests.post(API_URL, json=params, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API error {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "❌ Cannot connect to Flask server. Make sure app.py is running."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "❌ Request timed out."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_payload(budget_usd, popularity, avg_vote, vote_count, runtime,
                  release_month, release_year, genres, is_english, is_adult,
                  has_tagline, num_cast, num_directors, num_writers,
                  num_prod_cos, num_spoken_langs):
    return {
        "budget_usd": budget_usd,
        "popularity": popularity,
        "avg_vote": avg_vote,
        "vote_count": vote_count,
        "num_votes": vote_count,
        "runtime": runtime,
        "release_month": release_month,
        "release_year": release_year,
        "release_day": 15,
        "is_english": is_english,
        "is_adult": is_adult,
        "has_tagline": has_tagline,
        "genres": genres,
        "num_cast": num_cast,
        "num_directors": num_directors,
        "num_writers": num_writers,
        "num_production_companies": num_prod_cos,
        "num_spoken_languages": num_spoken_langs,
        "lang_freq": 3000 if is_english else 200,
    }


def call_claude_chat(user_message: str, prediction_context: dict) -> str:
    """Call Claude API for context-aware chat about the prediction."""
    try:
        context_str = ""
        if prediction_context:
            p = prediction_context
            context_str = f"""
You are an expert film investment analyst AI assistant integrated into Cinecast, a movie investment intelligence platform.

CURRENT FILM ANALYSIS CONTEXT:
- Verdict: {"GREENLIGHT ✅" if p['is_success'] else "FAIL ❌"}
- Model Confidence: {p['confidence']}% probability of commercial success
- Production Budget: ${p['budget_m']:.1f}M
- Expected IMDb Rating: {p['avg_vote']:.1f}/10
- IMDb Vote Count (audience engagement): {p['vote_count']:,}
- Release Month: {p['month_name']}
- Release Year: {p['release_year']}
- Genres: {', '.join(p['genres']) if p['genres'] else 'Not specified'}
- Language: {"English" if p['is_english'] else "Non-English"}
- Top 5 prediction drivers (model feature importances): {', '.join(p['top_features'])}

Answer in a professional, concise investment-analyst tone. Be direct and actionable.
Do not use excessive formatting. Keep responses to 3-5 sentences max unless more detail is requested.
Do not repeat the full context back — only reference what's relevant to the question.
"""
        else:
            context_str = """
You are an expert film investment analyst AI assistant integrated into Cinecast, a movie investment intelligence platform.
No prediction has been made yet. Answer general questions about film investment, the model, or how to use the platform.
Be professional, concise, and direct. 3-5 sentences max.
"""

        # Build conversation history for Gemini
        # Gemini uses "user" and "model" roles (not "assistant")
        raw_history = st.session_state.chat_history[-8:]
        gemini_history = []
        for msg in raw_history:
            role = "model" if msg["role"] == "assistant" else "user"
            # Skip consecutive duplicate roles
            if gemini_history and gemini_history[-1]["role"] == role:
                continue
            gemini_history.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        # Always end with the current user message
        if gemini_history and gemini_history[-1]["role"] == "user":
            gemini_history[-1] = {"role": "user", "parts": [{"text": user_message}]}
        else:
            gemini_history.append({"role": "user", "parts": [{"text": user_message}]})

        # Gemini REST endpoint (gemini-2.0-flash is fast and free-tier friendly)
        api_key = st.secrets["GOOGLE_API_KEY"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )

        payload = {
            "system_instruction": {"parts": [{"text": context_str}]},
            "contents": gemini_history,
            "generationConfig": {
                "maxOutputTokens": 1000,
                "temperature": 0.7,
            }
        }

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            error_detail = ""
            try:
                error_detail = response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            return f"I'm unable to respond right now (API status {response.status_code}: {error_detail}). Please try again."
    except Exception as e:
        return f"Chat unavailable: {str(e)}"


# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">CINECAST</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Movie Investment Intelligence</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider-gold">', unsafe_allow_html=True)


# =============================================================================
# EXAMPLE SCENARIOS
# =============================================================================
st.markdown('<div class="section-header">🎥 Quick Scenarios</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.9rem;'>"
    "Load a pre-filled example scenario, or fill in your own film details below.</p>",
    unsafe_allow_html=True
)

ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
for col, (name, ex) in zip([ex_col1, ex_col2, ex_col3, ex_col4], EXAMPLES.items()):
    with col:
        st.markdown(f"""
        <div class="example-card">
            <div class="example-title">{name}</div>
            <div class="example-meta">{ex['desc']}</div>
            <div class="example-meta" style="color:#d4af37; margin-top:0.4rem;">
                ${ex['budget_millions']}M · {MONTH_NAMES[ex['release_month']]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Example", key=f"ex_{name}"):
            st.session_state["run_example"] = ex
            st.session_state["example_name"] = name
            st.rerun()

st.markdown('<hr class="divider-gold">', unsafe_allow_html=True)


# =============================================================================
# STEP-BY-STEP INPUT FORM
# =============================================================================

# ── STEP 1: FINANCIAL ────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
    <span class="step-number">1</span>💰 Financial Details
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.88rem; margin-bottom:1rem;'>"
    "Enter the financial profile of your film. Budget is the primary ROI driver in our model.</p>",
    unsafe_allow_html=True
)

budget_millions = st.slider(
    "Production Budget (USD Millions)",
    min_value=0.1, max_value=400.0, value=50.0, step=0.5,
    help="Total production budget in USD millions. Does not include marketing/P&A spend."
)
budget_usd = budget_millions * 1_000_000

st.markdown("<br>", unsafe_allow_html=True)

# ── STEP 2: AUDIENCE ENGAGEMENT ──────────────────────────────────────────────
st.markdown("""
<div class="step-header">
    <span class="step-number">2</span>📊 Audience Engagement
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.88rem; margin-bottom:1rem;'>"
    "Audience engagement signals are the strongest predictors of commercial success in our model.</p>",
    unsafe_allow_html=True
)

a_col1, a_col2 = st.columns(2)
with a_col1:
    popularity = st.slider(
        "TMDB Popularity Score",
        min_value=1.0, max_value=300.0, value=25.0, step=1.0,
        help="TMDB popularity score. Blockbusters often exceed 100. Most films are 5–50."
    )
    avg_vote = st.slider(
        "Expected IMDb Rating (0–10)",
        min_value=1.0, max_value=10.0, value=6.5, step=0.1,
        help="Expected or current average IMDb user rating."
    )
with a_col2:
    vote_count = st.number_input(
        "IMDb Vote Count",
        min_value=10, max_value=2_000_000, value=5000, step=100,
        help="Total number of IMDb user votes. This is the single strongest predictor in the model."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── STEP 3: PRODUCTION & RELEASE ─────────────────────────────────────────────
st.markdown("""
<div class="step-header">
    <span class="step-number">3</span>🎥 Production & Release
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.88rem; margin-bottom:1rem;'>"
    "Production scale, genre, and release timing all influence the commercial outcome.</p>",
    unsafe_allow_html=True
)

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    runtime = st.slider("Runtime (minutes)", min_value=40, max_value=240, value=110, step=5)
    release_month = st.selectbox(
        "Release Month",
        options=list(range(1, 13)),
        index=5,
        format_func=lambda x: MONTH_NAMES[x],
        help="Summer (Jun–Aug) and Holiday (Nov–Dec) windows historically outperform."
    )
    release_year = st.slider("Release Year", min_value=1980, max_value=2026, value=2024, step=1)

with p_col2:
    genres_selected = st.multiselect(
        "Genres",
        options=["Action", "Drama", "Comedy", "Thriller", "Adventure",
                 "Science Fiction", "Romance", "Horror", "Animation", "Crime"],
        default=["Action", "Adventure"]
    )
    language = st.radio("Original Language", options=["English", "Other"], index=0, horizontal=True)
    is_english = 1 if language == "English" else 0

    is_adult_input = st.radio("Adult Content", options=["No", "Yes"], index=0, horizontal=True)
    is_adult = 1 if is_adult_input == "Yes" else 0

    has_tagline_input = st.radio("Has Marketing Tagline?", options=["Yes", "No"], index=0, horizontal=True)
    has_tagline = 1 if has_tagline_input == "Yes" else 0

with p_col3:
    num_cast = st.slider("Billed Cast Members", min_value=1, max_value=30, value=8)
    num_directors = st.slider("Number of Directors", min_value=1, max_value=5, value=1)
    num_writers = st.slider("Number of Writers", min_value=0, max_value=8, value=2)
    num_prod_cos = st.slider("Production Companies", min_value=1, max_value=10, value=2,
                              help="More companies often signals wider distribution.")
    num_spoken_langs = st.slider("Spoken Languages", min_value=1, max_value=8, value=1)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
predict_clicked = st.button("🎬 GENERATE INVESTMENT VERDICT", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# HANDLE PREDICTION
# =============================================================================
result = None
input_params = None
example_name = None

if predict_clicked:
    input_params = build_payload(
        budget_usd, popularity, avg_vote, vote_count, runtime,
        release_month, release_year, genres_selected, is_english, is_adult,
        has_tagline, num_cast, num_directors, num_writers, num_prod_cos, num_spoken_langs
    )
    with st.spinner("🎬 Analysing your film..."):
        result = call_api(input_params)

elif "run_example" in st.session_state:
    ex = st.session_state.pop("run_example")
    example_name = st.session_state.pop("example_name", "")
    ex_is_english = 1 if ex["language"] == "English" else 0
    input_params = build_payload(
        ex["budget_millions"] * 1_000_000,
        ex["popularity"], ex["avg_vote"], ex["vote_count"],
        ex["runtime"], ex["release_month"], ex["release_year"],
        ex["genres"], ex_is_english, ex["is_adult"],
        ex["has_tagline"], ex["num_cast"], ex["num_directors"],
        ex["num_writers"], ex["num_prod_cos"], ex["num_spoken_langs"]
    )
    with st.spinner(f"🎬 Evaluating {example_name}..."):
        result = call_api(input_params)


# =============================================================================
# DISPLAY RESULTS
# =============================================================================
if result is not None:

    st.markdown('<hr class="divider-gold">', unsafe_allow_html=True)

    if not result["success"]:
        st.error(result["error"])
        st.info("💡 Make sure `python3 app.py` is running in a separate terminal.")

    else:
        data         = result["data"]
        prediction   = data["prediction"]
        confidence   = data["confidence"]
        top_features = data["top_features"]
        is_success   = prediction == "Successful"

        budget_m    = input_params["budget_usd"] / 1_000_000
        month_name  = MONTH_NAMES.get(input_params["release_month"], "Unknown")
        genre_str   = ", ".join(input_params["genres"]) if input_params["genres"] else "Unspecified"
        lang_str    = "English" if input_params["is_english"] == 1 else "Non-English"
        feat_labels = [FEATURE_DISPLAY_NAMES.get(f["feature"], f["feature"]) for f in top_features]

        # Store context for chat
        st.session_state.last_prediction = {
            "is_success":   is_success,
            "confidence":   confidence,
            "budget_m":     budget_m,
            "avg_vote":     input_params["avg_vote"],
            "vote_count":   input_params["vote_count"],
            "month_name":   month_name,
            "release_year": input_params["release_year"],
            "genres":       input_params["genres"],
            "is_english":   input_params["is_english"],
            "top_features": feat_labels,
        }

        if example_name:
            st.markdown(
                f"<p style='font-family:Cinzel,serif; color:#9a8a7a; font-size:1rem;'>Evaluating: {example_name}</p>",
                unsafe_allow_html=True
            )

        # ── VERDICT CARD ─────────────────────────────────────────────────────
        verdict_class = "verdict-greenlight" if is_success else "verdict-fail"
        text_class    = "verdict-text-green" if is_success else "verdict-text-red"
        emoji         = "✅" if is_success else "❌"
        verdict_word  = "GREENLIGHT" if is_success else "FAIL"

        st.markdown(f"""
        <div class="{verdict_class}">
            <div class="{text_class}">{emoji} {verdict_word}</div>
            <div class="confidence-label">
                Model Confidence: <strong>{confidence}%</strong> probability of commercial success
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE BAR ────────────────────────────────────────────────────
        if confidence >= 70:
            bar_color, bar_label = "#2ecc71", "HIGH CONFIDENCE"
        elif confidence >= 50:
            bar_color, bar_label = "#f39c12", "MODERATE CONFIDENCE"
        else:
            bar_color, bar_label = "#e74c3c", "LOW CONFIDENCE"

        fig_conf = go.Figure(go.Bar(
            x=[confidence], y=["Confidence"], orientation="h",
            marker=dict(color=bar_color, line=dict(color="#d4af37", width=1)),
            text=[f"{confidence}% — {bar_label}"],
            textposition="inside",
            textfont=dict(size=13, color="white", family="Inter"),
            width=0.4,
        ))
        fig_conf.add_shape(type="line", x0=50, x1=50, y0=-0.5, y1=0.5,
                           line=dict(color="#d4af37", width=2, dash="dash"))
        fig_conf.update_layout(
            xaxis=dict(range=[0, 100], showgrid=False,
                       tickvals=[0, 25, 50, 75, 100],
                       ticktext=["0%", "25%", "50% (threshold)", "75%", "100%"],
                       tickfont=dict(color="#9a8a7a", size=11)),
            yaxis=dict(showticklabels=False, showgrid=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,10,5,0.8)",
            height=110, margin=dict(l=10, r=10, t=10, b=30),
            font=dict(color="#e8e0d0"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # ── EXECUTIVE INVESTMENT BRIEF ────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Executive Investment Brief</div>',
                    unsafe_allow_html=True)

        # Determine risk tier
        if confidence >= 80:
            risk_tier = "Low Risk" if is_success else "High Risk"
            risk_color = "#2ecc71" if is_success else "#e74c3c"
        elif confidence >= 60:
            risk_tier = "Moderate Risk"
            risk_color = "#f39c12"
        else:
            risk_tier = "High Risk" if is_success else "Moderate Risk"
            risk_color = "#e74c3c" if is_success else "#f39c12"

        # Release window assessment
        summer_months = [6, 7, 8]
        holiday_months = [11, 12]
        award_months = [9, 10, 1, 2]
        if input_params["release_month"] in summer_months:
            window_assessment = "Summer blockbuster window — historically the highest-grossing theatrical period"
            window_positive = True
        elif input_params["release_month"] in holiday_months:
            window_assessment = "Holiday release window — strong family and awards-contender performance"
            window_positive = True
        elif input_params["release_month"] in award_months:
            window_assessment = "Awards season window — suitable for prestige titles, limited wide-release upside"
            window_positive = None
        else:
            window_assessment = "Off-peak release window — reduced competition but lower theatrical footprint expected"
            window_positive = False

        # Genre ROI context
        high_roi_genres = ["Horror", "Thriller", "Animation"]
        high_budget_genres = ["Action", "Adventure", "Science Fiction"]
        genre_flags_active = [g for g in (input_params["genres"] or []) if g in high_roi_genres]
        genre_budget_active = [g for g in (input_params["genres"] or []) if g in high_budget_genres]

        if genre_flags_active:
            genre_note = f"Genre mix includes {', '.join(genre_flags_active)} — historically high ROI relative to budget in this model's training data."
        elif genre_budget_active:
            genre_note = f"Genre mix ({', '.join(genre_budget_active)}) aligns with high-budget commercial releases. Audience scale expectations should match production investment."
        else:
            genre_note = f"Genre mix ({genre_str}) is positioned outside both the high-ROI and high-budget clusters. Differentiated positioning will be key to market performance."

        # Engagement vs Budget signal
        vote_budget_signal = ""
        if input_params["vote_count"] >= 100_000:
            vote_budget_signal = "Audience engagement volume is at franchise-level scale — a strong positive signal for the model."
        elif input_params["vote_count"] >= 20_000:
            vote_budget_signal = "Audience engagement is in the mid-tier range. Increasing pre-release marketing investment could move this metric and improve the prediction."
        else:
            vote_budget_signal = f"Audience engagement (vote count: {input_params['vote_count']:,}) is below the dataset median. This is the model's most sensitive feature — a lower count significantly suppresses success probability."

        # Main verdict paragraph
        if is_success:
            main_verdict = (
                f"Based on the parameters provided, this project clears the commercial viability threshold with "
                f"<strong>{confidence}% confidence</strong>. "
                f"The model places this film in the <strong>Greenlight</strong> category — "
                f"meaning revenue is predicted to exceed production budget. "
                f"Risk classification: <span style='color:{risk_color};'><strong>{risk_tier}</strong></span>."
            )
        else:
            main_verdict = (
                f"Based on the parameters provided, this project does not clear the commercial viability threshold. "
                f"The model places this film in the <strong>Fail</strong> category with "
                f"<strong>{confidence}% probability</strong> that revenue will fall below production budget. "
                f"Risk classification: <span style='color:{risk_color};'><strong>{risk_tier}</strong></span>."
            )

        st.markdown(f"""
        <div class="intel-box">
            <h4>🎯 Verdict Summary</h4>
            <p>{main_verdict}</p>
            <p>{vote_budget_signal}</p>
            <p>{genre_note}</p>
            <p>Release window: <strong>{month_name} {input_params['release_year']}</strong> — {window_assessment}.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── WHAT'S DRIVING THIS DECISION ─────────────────────────────────────
        st.markdown('<div class="section-header">🔍 What is Driving This Decision</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.88rem; margin-bottom:0.8rem;'>
        The chart below shows the five most influential factors in our Gradient Boosting model globally.
        These factors consistently explain the largest share of variance across all predictions —
        not just yours. Use this to understand <em>where to focus</em> when adjusting parameters.
        </p>
        """, unsafe_allow_html=True)

        feat_values = [f["importance"] for f in top_features]

        fig_feat = go.Figure(go.Bar(
            x=feat_values[::-1], y=feat_labels[::-1], orientation="h",
            marker=dict(
                color=["#d4af37", "#c8a030", "#b89020", "#a88010", "#987000"],
                line=dict(color="#8b1a1a", width=1)
            ),
            text=[f"{v:.4f}" for v in feat_values[::-1]],
            textposition="outside",
            textfont=dict(size=12, color="#d4af37"),
        ))
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,10,5,0.8)",
            height=260, margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#2a1a0a",
                       tickfont=dict(color="#9a8a7a", size=10),
                       title=dict(text="Feature Importance Score", font=dict(color="#9a8a7a", size=11))),
            yaxis=dict(tickfont=dict(color="#e8e0d0", size=12), showgrid=False),
            font=dict(color="#e8e0d0"),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        # How each driver relates to this film
        driver_interpretations = []
        for feat, label in zip([f["feature"] for f in top_features], feat_labels):
            if "Vote" in label or "Engagement" in label:
                val = input_params["vote_count"]
                if val >= 100_000:
                    interp = f"<strong>{label}:</strong> Your vote count of {val:,} is franchise-scale — a strong positive signal."
                elif val >= 20_000:
                    interp = f"<strong>{label}:</strong> Your vote count of {val:,} is mid-tier. Consider investing in pre-release audience building."
                else:
                    interp = f"<strong>{label}:</strong> Your vote count of {val:,} is low. This is the model's most sensitive lever — increasing it materially improves success odds."
            elif "Budget" in label:
                interp = f"<strong>{label}:</strong> ${budget_m:.1f}M sits {'above' if budget_m > 50 else 'below'} the dataset median (~$50M). {'Higher budget correlates with broader distribution infrastructure.' if budget_m > 50 else 'Lower budget reduces risk exposure but also caps expected revenue ceiling.'}"
            elif "Age" in label or "Year" in label:
                age = 2024 - input_params["release_year"]
                interp = f"<strong>{label}:</strong> A release year of {input_params['release_year']} gives a film age of {age} years in the model. Newer releases have limited historical data but align with current audience preferences."
            elif "Rating" in label or "Vote Quality" in label:
                interp = f"<strong>{label}:</strong> An expected IMDb rating of {input_params['avg_vote']:.1f}/10 places this film in the {'top-tier' if input_params['avg_vote'] >= 7.5 else 'mid-tier' if input_params['avg_vote'] >= 6.0 else 'lower-tier'} quality bracket."
            else:
                interp = f"<strong>{label}:</strong> A secondary factor in this prediction. Review genre and release timing to optimise this signal."
            driver_interpretations.append(interp)

        interp_html = "".join(
            f'<div class="action-item"><span class="action-bullet">▸</span><span>{item}</span></div>'
            for item in driver_interpretations
        )
        st.markdown(f'<div class="intel-box"><h4>📌 How These Drivers Apply to Your Film</h4>{interp_html}</div>',
                    unsafe_allow_html=True)

        # ── PRODUCER RECOMMENDATIONS ─────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Producer Recommendations</div>',
                    unsafe_allow_html=True)

        recommendations = []

        if is_success:
            if confidence < 70:
                recommendations.append("Confidence is moderate. Before greenlighting, stress-test the budget model — small overruns could flip the ROI.")
            if input_params["release_month"] not in [5, 6, 7, 8, 11, 12]:
                recommendations.append(f"Consider shifting the release to a summer or holiday window. The current {month_name} slot is off-peak and may suppress theatrical returns.")
            if input_params["vote_count"] < 20_000:
                recommendations.append("Invest in pre-release audience engagement campaigns (trailers, social, screenings). IMDb vote count is the model's top driver — early buzz directly moves this number.")
            if not input_params["genres"]:
                recommendations.append("Define the genre positioning clearly before marketing spend. Genre signals influence distributor placement and audience targeting.")
            if budget_m > 100 and input_params["vote_count"] < 50_000:
                recommendations.append(f"The budget-to-engagement ratio is a concern: ${budget_m:.0f}M production with only {input_params['vote_count']:,} votes. This gap historically correlates with underperformance for big-budget titles.")
            if not recommendations:
                recommendations.append("Film profile is strong across key metrics. Maintain production quality and execute a broad theatrical release strategy.")
                recommendations.append("Prioritise franchise potential or sequel rights in deal structuring — the profile aligns with audience-scale titles.")
        else:
            if input_params["vote_count"] < 10_000:
                recommendations.append(f"The single highest-impact change: increase audience engagement. At {input_params['vote_count']:,} votes, the model assigns very low success probability. A multi-platform pre-release campaign targeting 50K+ votes would materially shift the prediction.")
            if budget_m > 80:
                recommendations.append(f"Consider reducing the production budget. At ${budget_m:.0f}M, the revenue-to-budget ratio required for profitability is high given current engagement signals. A revised budget of $30–50M may achieve a Greenlight verdict.")
            if input_params["release_month"] not in [5, 6, 7, 8, 11, 12]:
                recommendations.append(f"Shift the release window. {month_name} is an off-peak month. A summer or holiday release would increase success probability independently of other changes.")
            if not input_params["genres"]:
                recommendations.append("Anchor the film to a genre with historically strong ROI (Horror, Thriller, Animation) if creatively feasible. Genre signals affect the model's posterior probability.")
            if not recommendations:
                recommendations.append("The model's Fail verdict reflects a combination of factors. Run multiple scenarios adjusting budget, engagement, and release timing separately to identify the smallest change that achieves Greenlight status.")

        recs_html = "".join(
            f'<div class="action-item"><span class="action-bullet">▸</span><span>{r}</span></div>'
            for r in recommendations
        )
        rec_header = "🟢 Optimisation Opportunities" if is_success else "🔴 Critical Risk Factors & Remediation"
        st.markdown(f'<div class="intel-box"><h4>{rec_header}</h4>{recs_html}</div>',
                    unsafe_allow_html=True)

        # ── RISK FLAGS ────────────────────────────────────────────────────────
        risk_flags = []
        if confidence < 65 and is_success:
            risk_flags.append(f"⚠️ Borderline Greenlight — confidence is only {confidence}%. A 5–10% swing in key parameters (budget overrun, lower engagement) could flip this to a Fail verdict.")
        if budget_m > 150 and input_params["vote_count"] < 100_000:
            risk_flags.append(f"⚠️ High-Budget / Low-Engagement gap — ${budget_m:.0f}M production with {input_params['vote_count']:,} votes is a structural risk. These films historically carry the highest write-off risk.")
        if not input_params["is_english"] and budget_m > 30:
            risk_flags.append("⚠️ Non-English language film with significant budget — international distribution infrastructure will need to be secured explicitly.")
        if input_params["is_adult"]:
            risk_flags.append("⚠️ Adult content flag is active — theatrical distribution in major markets will be restricted. Revenue ceiling is significantly lower.")

        if risk_flags:
            for flag in risk_flags:
                st.markdown(f'<div class="risk-flag">{flag}</div>', unsafe_allow_html=True)

        # ── INPUT SNAPSHOT ────────────────────────────────────────────────────
        with st.expander("📂 View Full Input Summary", expanded=False):
            s_col1, s_col2, s_col3 = st.columns(3)
            snapshots = [
                (f"${budget_m:.1f}M", "Production Budget"),
                (f"{input_params['avg_vote']:.1f}/10", "Expected IMDb Rating"),
                (f"{input_params['vote_count']:,}", "IMDb Vote Count"),
                (genre_str, "Genres"),
                (month_name, "Release Month"),
                (lang_str, "Language"),
            ]
            for i, (val, lbl) in enumerate(snapshots):
                with [s_col1, s_col2, s_col3][i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.2rem;">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)


# =============================================================================
# AI CHATBOX — SECTION BELOW RESULTS
# =============================================================================
st.markdown('<hr class="divider-gold">', unsafe_allow_html=True)
st.markdown('<div class="section-header">💬 Investment Intelligence Chat</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:Inter,sans-serif; color:#9a8a7a; font-size:0.88rem; margin-bottom:1rem;'>"
    "Ask any questions about this film's prediction, investment strategy, or general industry context. "
    "The assistant is context-aware and knows your current results.</p>",
    unsafe_allow_html=True
)

# Suggested starter questions (only show if prediction exists)
if st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    starters = [
        "What's the biggest risk in this film's profile?",
        "How could I change this to a Greenlight?" if not pred["is_success"] else "How do I maximise returns from here?",
        "Which genre would give the best ROI at this budget?",
        "What release strategy do you recommend?",
        "How does this compare to a typical successful film in this dataset?",
    ]
    st.markdown(
        "<p style='font-family:Inter,sans-serif; color:#7a6a6a; font-size:0.8rem; margin-bottom:0.5rem;'>"
        "Suggested questions:</p>",
        unsafe_allow_html=True
    )
    sug_cols = st.columns(len(starters))
    for i, (col, q) in enumerate(zip(sug_cols, starters)):
        with col:
            if st.button(q, key=f"sug_{i}", help=q):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    reply = call_claude_chat(q, st.session_state.last_prediction)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

# Display chat history
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label-user">YOU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label-ai">CINECAST AI</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Chat input
chat_col1, chat_col2 = st.columns([5, 1])
with chat_col1:
    user_chat_input = st.text_input(
        "Ask a question about your film or investment strategy...",
        key="chat_text_input",
        label_visibility="collapsed",
        placeholder="e.g. What would happen if I increased the budget to $100M?"
    )
with chat_col2:
    send_btn = st.button("Send →", use_container_width=True)

if send_btn and user_chat_input.strip():
    question = user_chat_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        reply = call_claude_chat(question, st.session_state.last_prediction)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# =============================================================================
# DISCLAIMER
# =============================================================================
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Academic Disclaimer</strong><br>
    Cinecast is an ML-based investment intelligence tool developed as a capstone project for the
    PGP Data Science with GenAI programme at Great Lakes Institute of Management (September 2025 cohort).
    Predictions are generated by a Gradient Boosting classifier trained on historical TMDB + IMDb data
    and are intended for academic demonstration purposes only.
    They do not constitute financial, investment, or production advice.<br><br>
    Group 6: Ms. Manal Munawwar | Mr. A Sivaranjan | Ms. Poovizhi P |
    Mr. Saran | Ms. Vijayalakshmi | Mentor: Mr. Aishwarya Sarda
</div>
""", unsafe_allow_html=True)