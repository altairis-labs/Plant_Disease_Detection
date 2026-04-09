"""
disease_info.py
---------------
Lookup table for all 38 PlantVillage classes.
Returns cure, sunlight, watering, and nutrient advice + severity level.

severity values: "none" | "moderate" | "severe"
"""

_DB = {
    # ── APPLE ────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "severity": "moderate",
        "cure": (
            "Remove and destroy all infected leaves and fruit. Apply fungicides "
            "containing captan, myclobutanil, or mancozeb at 7–10 day intervals "
            "starting at green tip stage. Prune for good air circulation."
        ),
        "sun":       "Full sun (6–8 hrs/day). Good light reduces leaf wetness duration.",
        "water":     "Water at the base; avoid wetting foliage. Drip irrigation preferred.",
        "nutrients": "Balanced NPK (10-10-10). Avoid excess nitrogen which promotes soft tissue.",
    },
    "Apple___Black_rot": {
        "severity": "severe",
        "cure": (
            "Prune out all dead or cankered wood at least 15 cm below visible infection. "
            "Apply captan or thiophanate-methyl fungicide. Remove mummified fruit."
        ),
        "sun":       "Full sun. Open canopy slows moisture buildup and fungal spread.",
        "water":     "Reduce overhead watering. Ensure soil drains well to avoid root stress.",
        "nutrients": "Potassium-rich fertiliser supports cell wall strength and disease resistance.",
    },
    "Apple___Cedar_apple_rust": {
        "severity": "moderate",
        "cure": (
            "Apply myclobutanil or propiconazole fungicide from pink bud stage through petal fall. "
            "Remove nearby cedar/juniper trees if possible as they are the alternate host."
        ),
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "Avoid overhead irrigation. Water early in the morning.",
        "nutrients": "Moderate nitrogen; excess promotes susceptible succulent growth.",
    },
    "Apple___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Continue regular monitoring and preventive care.",
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "1–2 inches per week; consistent moisture, especially during fruit set.",
        "nutrients": "Balanced NPK in spring; potassium supplement at fruit development.",
    },

    # ── BLUEBERRY ─────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Maintain acidic soil (pH 4.5–5.5).",
        "sun":       "Full sun to light partial shade.",
        "water":     "1–2 inches/week. Blueberries are shallow-rooted; consistent moisture critical.",
        "nutrients": "Acid-forming fertiliser (ammonium sulfate). Avoid phosphorus excess.",
    },

    # ── CHERRY ────────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "moderate",
        "cure": (
            "Apply sulfur-based or potassium bicarbonate fungicide at first sign. "
            "Remove infected shoot tips. Improve air circulation by pruning."
        ),
        "sun":       "Full sun. Shaded, humid conditions worsen powdery mildew.",
        "water":     "Avoid wetting leaves. Water at base in the morning.",
        "nutrients": "Reduce nitrogen — excessive growth creates susceptible soft tissue.",
    },
    "Cherry_(including_sour)___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Scout weekly for early signs of mildew or leaf spot.",
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "Regular deep watering; reduce frequency as harvest approaches.",
        "nutrients": "Balanced NPK in early spring; light potassium feed mid-season.",
    },

    # ── CORN ──────────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "moderate",
        "cure": (
            "Plant resistant hybrids. Apply strobilurin or triazole fungicides at early "
            "disease onset. Rotate crops — avoid continuous corn."
        ),
        "sun":       "Full sun. Dense plantings increase humidity; maintain spacing.",
        "water":     "Avoid prolonged leaf wetness. Furrow or drip irrigation preferred.",
        "nutrients": "Adequate potassium and phosphorus improve resistance. Avoid excess N.",
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "moderate",
        "cure": (
            "Plant resistant varieties. Apply triazole fungicides (propiconazole) if infection "
            "is severe before tasseling. Scout fields weekly."
        ),
        "sun":       "Full sun.",
        "water":     "Overhead irrigation promotes spore spread; use furrow irrigation if possible.",
        "nutrients": "Balanced fertilisation; potassium improves disease tolerance.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "severe",
        "cure": (
            "Use resistant hybrids. Apply azoxystrobin or propiconazole fungicide at first sign. "
            "Bury or remove infected residue after harvest."
        ),
        "sun":       "Full sun.",
        "water":     "Reduce leaf wetness duration. Early morning irrigation only.",
        "nutrients": "Adequate nitrogen supports recovery; do not over-apply.",
    },
    "Corn_(maize)___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Continue crop rotation and weed management.",
        "sun":       "Full sun.",
        "water":     "1–1.5 inches/week; consistent moisture at silking stage is critical.",
        "nutrients": "High nitrogen demand — side-dress at V6 stage.",
    },

    # ── GRAPE ─────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "severity": "severe",
        "cure": (
            "Apply myclobutanil or mancozeb from bud break through early fruit development "
            "at 10-day intervals. Remove all mummified berries and infected canes."
        ),
        "sun":       "Full sun. Good light and air flow through the canopy are critical.",
        "water":     "Avoid wetting fruit and foliage. Drip irrigation strongly preferred.",
        "nutrients": "Potassium supports berry cell integrity. Avoid luxury nitrogen.",
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "severe",
        "cure": (
            "No chemical cure. Remove and burn infected wood. Prune in dry conditions; "
            "seal cuts with fungicidal wound paste. Replace severely affected vines."
        ),
        "sun":       "Full sun with good ventilation.",
        "water":     "Avoid water stress — it triggers symptom expression. Consistent irrigation.",
        "nutrients": "Balanced nutrition; potassium and calcium support vine structural health.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "moderate",
        "cure": (
            "Apply copper-based or mancozeb fungicides. Remove infected leaves. "
            "Improve air circulation by canopy management."
        ),
        "sun":       "Full sun.",
        "water":     "Overhead watering spreads spores — switch to drip.",
        "nutrients": "Balanced NPK; avoid excess nitrogen.",
    },
    "Grape___healthy": {
        "severity": "none",
        "cure":      "Vine is healthy. Maintain canopy management and preventive copper sprays.",
        "sun":       "Full sun (8+ hrs/day).",
        "water":     "Deep, infrequent watering. Drip irrigation preferred.",
        "nutrients": "Balanced NPK; potassium-rich at veraison (colour change).",
    },

    # ── ORANGE ────────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "severe",
        "cure": (
            "No cure exists. Remove and destroy infected trees immediately to prevent spread. "
            "Control Asian citrus psyllid (the insect vector) with systemic insecticides. "
            "Plant certified disease-free nursery stock."
        ),
        "sun":       "Full sun.",
        "water":     "Consistent irrigation; stressed trees show symptoms faster.",
        "nutrients": "Foliar micronutrient sprays (zinc, manganese) may delay symptom progression.",
    },

    # ── PEACH ─────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "severity": "moderate",
        "cure": (
            "Apply copper-based bactericides from shuck split through harvest at 7–14 day "
            "intervals. Plant resistant varieties. Avoid overhead irrigation."
        ),
        "sun":       "Full sun.",
        "water":     "Drip irrigation; avoid wetting fruit and leaves.",
        "nutrients": "Balanced NPK; adequate calcium reduces fruit susceptibility.",
    },
    "Peach___healthy": {
        "severity": "none",
        "cure":      "Tree is healthy. Apply dormant copper spray each year as prevention.",
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "Deep watering 1–2 times per week; reduce at harvest.",
        "nutrients": "Nitrogen in early spring; potassium and calcium mid-season.",
    },

    # ── PEPPER (BELL) ─────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "severity": "moderate",
        "cure": (
            "Apply copper hydroxide or copper octanoate bactericide weekly from transplant. "
            "Use certified disease-free seed. Rotate crops; avoid solanaceous family."
        ),
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "Drip irrigation; avoid overhead watering which spreads bacteria.",
        "nutrients": "Balanced NPK; calcium foliar spray reduces fruit infection.",
    },
    "Pepper,_bell___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Scout for aphids and thrips that vector viruses.",
        "sun":       "Full sun.",
        "water":     "Consistent moisture; avoid drought stress which cracks fruit.",
        "nutrients": "Moderate nitrogen; high potassium at fruiting stage.",
    },

    # ── POTATO ────────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "severity": "moderate",
        "cure": (
            "Apply chlorothalonil, mancozeb, or azoxystrobin at 7–10 day intervals. "
            "Remove infected lower leaves. Ensure adequate spacing for air flow."
        ),
        "sun":       "Full sun.",
        "water":     "Avoid wetting foliage. Irrigate at base in the morning.",
        "nutrients": "Adequate nitrogen; potassium improves disease tolerance.",
    },
    "Potato___Late_blight": {
        "severity": "severe",
        "cure": (
            "URGENT: Apply metalaxyl + mancozeb or cymoxanil fungicides immediately. "
            "Destroy infected haulm before harvest. This disease spreads extremely rapidly "
            "in cool, wet conditions — act within 24 hours of detection."
        ),
        "sun":       "Full sun. Cool, overcast conditions favour disease spread.",
        "water":     "Absolutely avoid overhead irrigation. Keep foliage dry.",
        "nutrients": "Avoid excess nitrogen; adequate potassium improves resistance.",
    },
    "Potato___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Apply preventive fungicide in wet weather periods.",
        "sun":       "Full sun.",
        "water":     "Consistent moisture; hill up soil to avoid tuber exposure.",
        "nutrients": "High potassium demand for tuber development.",
    },

    # ── RASPBERRY ─────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Prune out old floricanes after harvest.",
        "sun":       "Full sun to partial shade.",
        "water":     "1–2 inches/week; mulch to retain moisture and suppress weeds.",
        "nutrients": "Balanced NPK in early spring; avoid excess nitrogen after flowering.",
    },

    # ── SOYBEAN ───────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Monitor for sudden death syndrome and soybean cyst nematode.",
        "sun":       "Full sun.",
        "water":     "Consistent moisture at pod fill is critical for yield.",
        "nutrients": "Nitrogen fixed by rhizobia; ensure good nodulation. Adequate potassium.",
    },

    # ── SQUASH ────────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "severity": "moderate",
        "cure": (
            "Apply potassium bicarbonate, sulfur, or neem oil at first sign. "
            "Remove severely infected leaves. Plant resistant varieties."
        ),
        "sun":       "Full sun. Shade and poor air circulation dramatically worsen powdery mildew.",
        "water":     "Water at base only; wet foliage encourages fungal germination.",
        "nutrients": "Reduce nitrogen to slow the growth of succulent susceptible tissue.",
    },

    # ── STRAWBERRY ────────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "severity": "moderate",
        "cure": (
            "Apply myclobutanil or captan fungicide. Remove infected leaves. "
            "Avoid overhead watering. Renovate beds after harvest."
        ),
        "sun":       "Full sun (6–8 hrs/day).",
        "water":     "Drip irrigation; keep foliage dry.",
        "nutrients": "Balanced NPK; potassium supports disease resistance.",
    },
    "Strawberry___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Renovate beds annually; remove runners to maintain spacing.",
        "sun":       "Full sun.",
        "water":     "Consistent moisture; mulch with straw to reduce splash and fruit rot.",
        "nutrients": "Balanced NPK; potassium at runner formation and fruiting.",
    },

    # ── TOMATO ────────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "severity": "moderate",
        "cure": (
            "Apply copper-based bactericide at 7-day intervals. Remove infected tissue. "
            "Use certified disease-free transplants. Rotate out of solanaceous crops."
        ),
        "sun":       "Full sun (8 hrs/day).",
        "water":     "Drip only; bacteria spread rapidly in overhead irrigation water.",
        "nutrients": "Balanced NPK; calcium prevents blossom end rot (often co-occurring).",
    },
    "Tomato___Early_blight": {
        "severity": "moderate",
        "cure": (
            "Apply chlorothalonil or azoxystrobin fungicide. Remove infected lower leaves. "
            "Stake plants for air circulation. Rotate with non-solanaceous crops."
        ),
        "sun":       "Full sun.",
        "water":     "Water at base in the morning; avoid splashing soil onto leaves.",
        "nutrients": "Consistent balanced nutrition; nitrogen deficiency worsens susceptibility.",
    },
    "Tomato___Late_blight": {
        "severity": "severe",
        "cure": (
            "URGENT: Apply metalaxyl + mancozeb immediately. Remove and bag infected plants. "
            "Do not compost. Cool, wet weather accelerates spread to entire field within days."
        ),
        "sun":       "Full sun. Avoid growing in low-lying foggy areas.",
        "water":     "Drip irrigation only. Keep foliage completely dry.",
        "nutrients": "Potassium improves cell strength; avoid excess nitrogen.",
    },
    "Tomato___Leaf_Mold": {
        "severity": "moderate",
        "cure": (
            "Improve greenhouse ventilation. Apply chlorothalonil or copper fungicide. "
            "Remove infected leaves. Disease favours high humidity above 85%."
        ),
        "sun":       "Full sun; ensure adequate light penetration into canopy.",
        "water":     "Reduce humidity; water at base and increase ventilation.",
        "nutrients": "Balanced nutrition; avoid luxury nitrogen.",
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "moderate",
        "cure": (
            "Apply chlorothalonil, copper, or azoxystrobin fungicide. "
            "Remove infected leaves. Avoid working in fields when plants are wet."
        ),
        "sun":       "Full sun.",
        "water":     "Base watering only; mulch to prevent soil splash.",
        "nutrients": "Adequate calcium; balanced NPK.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "moderate",
        "cure": (
            "Apply miticide (abamectin, bifenazate) or insecticidal soap. Spray undersides "
            "of leaves. Introduce predatory mites (Phytoseiidae) for biological control. "
            "Mites thrive in hot, dry conditions."
        ),
        "sun":       "Full sun; monitor more intensely in dry, hot spells.",
        "water":     "Maintain adequate soil moisture; drought-stressed plants are more susceptible.",
        "nutrients": "Balanced nutrition; silica supplements can reduce mite feeding.",
    },
    "Tomato___Target_Spot": {
        "severity": "moderate",
        "cure": (
            "Apply azoxystrobin, chlorothalonil, or fluxapyroxad fungicide. "
            "Remove infected leaves. Improve spacing and air circulation."
        ),
        "sun":       "Full sun.",
        "water":     "Drip irrigation; avoid overhead watering.",
        "nutrients": "Balanced NPK; avoid over-fertilising with nitrogen.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "severe",
        "cure": (
            "No cure. Remove and destroy infected plants immediately. Control whitefly "
            "(the vector) with imidacloprid or reflective mulches. Plant resistant varieties. "
            "Use insect-proof nets in high-pressure areas."
        ),
        "sun":       "Full sun.",
        "water":     "Consistent moisture to reduce plant stress.",
        "nutrients": "Balanced nutrition; avoid luxury nitrogen which attracts whitefly.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "severe",
        "cure": (
            "No cure. Remove infected plants. Disinfect tools with 10% bleach solution. "
            "Wash hands before handling plants — virus spreads by contact. "
            "Plant resistant varieties (Tm-2² resistance gene)."
        ),
        "sun":       "Full sun.",
        "water":     "Avoid water stress; stressed plants show more severe symptoms.",
        "nutrients": "Balanced nutrition to support plant vigour.",
    },
    "Tomato___healthy": {
        "severity": "none",
        "cure":      "Plant is healthy. Scout weekly for early signs of blight or virus.",
        "sun":       "Full sun (8 hrs/day).",
        "water":     "Consistent 1–2 inches/week; mulch to maintain moisture.",
        "nutrients": "High potassium at fruiting; calcium prevents blossom end rot.",
    },
}


def get_disease_info(label: str) -> dict:
    """
    Return disease info dict for a CLASS_NAMES label.
    Falls back to a safe default if label not found.
    """
    if label in _DB:
        return _DB[label]

    # Try case-insensitive lookup
    for key, val in _DB.items():
        if key.lower() == label.lower():
            return val

    # Default fallback
    return {
        "severity": "unknown",
        "cure":      "No specific treatment data available. Consult a local agricultural extension service.",
        "sun":       "Refer to crop-specific guidelines.",
        "water":     "Refer to crop-specific guidelines.",
        "nutrients": "Refer to crop-specific guidelines.",
    }
