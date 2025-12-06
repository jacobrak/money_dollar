from fastapi import FastAPI
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform
import uvicorn

# --------------------------
# INIT VERTEX AI
# --------------------------
aiplatform.init(project="fixathon-480319", location="europe-west1")

app = FastAPI()
model = GenerativeModel("gemini-2.5-flash")

class Article(BaseModel):
    article: str | None = None

def read_article_file():
    with open("article.txt", "r", encoding="utf-8") as f:
        return f.read()


# --------------------------
# STAGE 1: GLOBAL + COMPANY ANALYSIS
# --------------------------
PERSPECTIVE_PROMPTS = {
    "global": """
You are a global macro-financial impact analyst.

Your job is to determine whether the news article has ANY meaningful
global geopolitical or economic impact.

Rules:
- If the event only affects a single company or a small sector → LOW impact.
- If the event affects global supply chains, energy, banks, shipping, or currencies → HIGH impact.
- If the article is commentary or non-economic → NO impact.

Output MUST include:

Impact_level: <none | low | moderate | high>
Impact_channels: <list of real mechanisms>
Explanation: <max 400 words>
Impact_score: <0–10>

ARTICLE:
""",

    "companies": """
You are a global corporate strategy analyst.

Analyze how private-sector companies will react to the events in the article.

Include:
- Supply chain changes
- Pricing decisions
- Hiring/layoffs
- Investment changes
- Sector-by-sector differences

Max 800 words.

ARTICLE:
"""
}

# --------------------------
# STAGE 2: ECONOMIC IMPACT
# --------------------------
ECON_IMPACT_PROMPT = """
You are a global economic transmission model.

You will receive:
1. A GLOBAL IMPACT ANALYSIS
2. A COMPANY REACTION ANALYSIS

Using these, produce a unified global economic impact in this JSON:

{
  "suppliers": "",
  "manufacturers": "",
  "retailers": "",
  "consumers": {
    "behavior": "",
    "demand_change": "<increase | decrease | neutral>",
    "why": ""
  },
  "risk_score": "<0-10>",
  "summary": ""
}

Risk score scale:
0 = no impact
10 = global systemic risk
"""

# --------------------------
# STAGE 3: SWEDISH SHOCK PREDICTOR
# --------------------------
SHOCK_PREDICTOR_PROMPT = """
You are a Swedish macroeconomic risk analyst at Riksbanken.

You will receive:
- A global economic impact assessment
- Current KPIF statistics
- The original news article

Your task:
Estimate how THIS SPECIFIC ARTICLE affects Swedish inflation risk.

Rules:
- If article is neutral → 0–35%
- If mixed → 35–60%
- If strongly negative → 60–95%
- Probability must vary — do NOT repeat previous values
- Base reasoning ONLY on article + impact JSON

Output JSON:

{
  "shock_probability": "<0-100%>",
  "direction": "<increase | decrease | neutral>",
  "mechanism": "",
  "key_drivers": ["", "", ""],
  "confidence_level": "<low | medium | high>",
  "article_sentiment": "<positive | neutral | negative>"
}
"""



# --------------------------
# MAIN PIPELINE
# --------------------------
@app.post("/analyze")
def analyze(a: Article = None):

    article_text = a.article if (a and a.article) else read_article_file()

    # Stage 1
    perspectives = {}
    for key, prompt in PERSPECTIVE_PROMPTS.items():
        response = model.generate_content(prompt + article_text)
        perspectives[key] = response.text

    # Stage 2
    combined_input = f"""
GLOBAL IMPACT ANALYSIS:
{perspectives['global']}

COMPANY ANALYSIS:
{perspectives['companies']}
"""
    econ_response = model.generate_content(
        ECON_IMPACT_PROMPT + "\n\n" + combined_input
    ).text

    # Stage 3


    shock_prompt = (
        SHOCK_PREDICTOR_PROMPT +
        "\n\nGLOBAL ECON IMPACT:\n" + econ_response +
        "\n\nGLOBAL AND COMPANY IMPACT" + combined_input +
        "\n\nARTICLE:\n" + article_text
    )

    shock_response = model.generate_content(shock_prompt).text

    return {
        "swedish_inflation_shock": shock_response
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
