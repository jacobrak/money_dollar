from fastapi import FastAPI
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform
import uvicorn
from typing import Optional
from fastapi import Body


# --------------------------
# INIT VERTEX AI (NECESSARY)
# --------------------------
# Make sure the region matches your project region, e.g. "us-central1" or "europe-west1"
aiplatform.init(project="fixathon-480319", location="europe-west1")

app = FastAPI()
model = GenerativeModel("gemini-2.5-flash")

class Article(BaseModel):
    article: str | None = None


# --------------------------
# 1. PROMPTS FOR STAGE 1
# --------------------------

def read_article_file():
    with open("article.txt", "r", encoding="utf-8") as f:
        return f.read()
    
PERSPECTIVE_PROMPTS = {
    "china": """
You are a Chinese government strategic analyst.
Analyze this article strictly from China's geopolitical and economic perspective.
Keep analysis to <800 words>.
""",
    "eu": """
You are a senior policy advisor for the European Union.
Analyze the article strictly from the EU's geopolitical, regulatory, and economic perspective.
Keep analysis to <800 words>.
""",
    "usa": """
You are a strategist for the United States National Security Council.
Analyze the article strictly from the U.S. security, economic, and diplomatic perspective.
Keep analysis to <800 words>.
"""
}


# --------------------------
# 2. STAGE 2 — GLOBAL IMPACT PROMPT
# --------------------------

ECON_IMPACT_PROMPT = """
You are a global economic impact model.

You will receive three geopolitical/economic analyses:
1. China perspective
2. EU perspective
3. USA perspective

Using all three perspectives, produce a unified economic impact assessment.

Your output MUST follow this exact JSON structure:

{
  "suppliers": "<explain behavior, risks, constraints, pricing, reactions>",
  "manufacturers": "<production changes, cost pressures, supply chain shifts>",
  "retailers": "<inventory, pricing strategy, margin pressure>",
  "consumers": {
    "behavior": "<what consumers do>",
    "demand_change": "<increase | decrease | neutral>",
    "why": "<explain reasoning>"
  },
  "risk_score": "<0-10>",
  "summary": "<short final summary>"
}

When calculating risk_score:
0 = no risk
10 = extremely high geopolitical + economic instability

Make sure to use evidence from all three regions.
"""
SWEDEN_IMPACT_PROMPT = """
You are a Swedish macroeconomic analyst at the Riksbank.

Using:
1. The China analysis
2. The EU analysis
3. The US analysis
4. The global economic impact JSON
5. Sweden’s economic structure:

- Very open economy (exports ≈ 50% of GDP)
- Deep integration with EU markets (53% of exports)
- Import-dependent manufacturing (raw materials, components)
- Highly interest-rate sensitive households (large mortgages)
- SEK is extremely sensitive to global risk sentiment and USD/EUR
- Energy and shipping costs strongly affect inflation
- Export industries: machinery, vehicles, metals

Estimate the impact on Sweden in this exact JSON format:

{
  "swedish_inflation_direction": "<increase | decrease | neutral>",
  "inflation_drivers": "<short explanation of what causes this>",
  "exchange_rate_effect": "<expected impact on SEK>",
  "export_industry_effect": "<short explanation>",
  "import_price_effect": "<short explanation>",
  "energy_price_effect": "<short explanation>",
  "riksbank_reaction": "<hawkish | dovish | unchanged>",
  "confidence_score": "<0-100>",
  "summary": "<short readable summary>"
}

Base your reasoning strictly on the geopolitical agents and Sweden's economic structure.
"""


# --------------------------
# 3. API PIPELINE
# --------------------------
# --------------------------
# 3. STAGE 3 — SWEDISH INFLATION SHOCK PREDICTION
# --------------------------

SHOCK_PREDICTOR_PROMPT = """
You are a Swedish macroeconomic risk analyst at Riksbanken.

You will receive two inputs:
1. A global economic impact assessment (JSON) based on China/EU/USA agent analysis.
2. Key Swedish inflation statistics (KPIF: current index, YoY rate, MoM rate, and trend).

Your task:
Estimate the probability that Sweden will experience an inflation shock within the next 3 months.

Definition of a Swedish inflation shock:
- A sudden unexpected change of ≥ 0.5 percentage points in KPIF within 3 months,
- Caused primarily by global transmission channels.

Transmission mechanisms you MUST evaluate:
- Import price pressure (energy, goods, food)
- Supply chain constraints affecting Swedish manufacturers
- Global demand effects on Swedish exports
- SEK exchange rate sensitivity to global risk sentiment
- Swedish consumer demand and confidence

Your output MUST be JSON:

{
  "shock_probability": "<0-100%>",
  "direction": "<increase | decrease | uncertain>",
  "mechanism": "<2-3 sentence causal explanation>",
  "key_drivers": [
    "<driver 1>",
    "<driver 2>",
    "<driver 3>"
  ],
  "confidence_level": "<low | medium | high>"
}

Be rigorous and base reasoning only on the impact JSON and Swedish inflation figures.
"""

def get_kpif_summary():
    """
    Static KPIF summary placeholder.
    Replace with SCB API call later if needed.
    """
    return {
        "current_kpif": 120.41,
        "yoy": 1.5,
        "mom": 0.2,
        "trend": "stable_slowing"
    }

@app.post("/analyze")
def analyze(a: Article = None):

    # --- Stage 1: get country perspectives ---
    perspectives = {}
    article_text = a.article if (a and a.article) else read_article_file()

    for region, prompt in PERSPECTIVE_PROMPTS.items():
        full_prompt = f"{prompt}\n\nARTICLE:\n{article_text}"
        response = model.generate_content(full_prompt)
        perspectives[region] = response.text

    # --- Stage 2: combine perspectives into econ impact ---
    combined_input = f"""
CHINA ANALYSIS:
{perspectives['china']}

EU ANALYSIS:
{perspectives['eu']}

USA ANALYSIS:
{perspectives['usa']}
"""

    final_prompt = ECON_IMPACT_PROMPT + "\n\n" + combined_input
    econ_response = model.generate_content(final_prompt).text

    # --- Stage 3: Swedish inflation shock prediction ---
    kpif_summary = get_kpif_summary()

    shock_prompt = (
        SHOCK_PREDICTOR_PROMPT +
        "\n\nGLOBAL ECONOMIC IMPACT:\n" + econ_response +
        "\n\nSWEDEN KPIF SUMMARY:\n" + str(kpif_summary)
    )

    shock_response = model.generate_content(shock_prompt).text

    # Return everything
    return {
        "swedish_inflation_shock": shock_response
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
