"""
disease_info.py — Treatment, cure, and care data for all 38 PlantVillage classes.
Used by plant_app.py to render the Treatment & Care tab.
"""

DISEASE_INFO = {

    # ── APPLE ──────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "status": "diseased",
        "cure": (
            "Apply fungicides containing captan, myclobutanil, or mancozeb at 7–10 day "
            "intervals during wet spring weather. Remove and destroy all fallen infected "
            "leaves in autumn — do not compost them. Prune for open canopy to improve air "
            "circulation. In severe cases, use copper-based sprays before bud break."
        ),
        "sun": "Full sun, minimum 6–8 hrs/day. Good air circulation reduces humidity and fungal spread.",
        "water": (
            "Water at the base only. Avoid wetting foliage. Morning watering is best — "
            "leaves dry out during the day. Drip irrigation strongly preferred."
        ),
        "nutrients": (
            "Balanced NPK (10-10-10) in early spring. Avoid excess nitrogen — it promotes "
            "lush soft tissue that is highly susceptible to scab infection. Add potassium "
            "to strengthen cell walls."
        ),
        "severity": "moderate",
    },

    "Apple___Black_rot": {
        "status": "diseased",
        "cure": (
            "Prune out all dead or cankered wood 8–10 inches below visible infection and "
            "destroy immediately. Apply captan or thiophanate-methyl fungicide from green "
            "tip through petal fall. Remove mummified fruits on the tree and ground. "
            "Disinfect pruning tools with 10% bleach between cuts."
        ),
        "sun": "Full sun. Avoid shaded, humid positions — they worsen rot spread.",
        "water": (
            "Keep foliage dry. Avoid overhead irrigation. Water deeply but infrequently "
            "at the root zone, especially during fruit development."
        ),
        "nutrients": (
            "Moderate nitrogen. Boost calcium (calcium nitrate spray) to improve fruit "
            "skin integrity. Ensure adequate boron for cell wall strength."
        ),
        "severity": "severe",
    },

    "Apple___Cedar_apple_rust": {
        "status": "diseased",
        "cure": (
            "Remove nearby eastern red cedar or juniper trees if possible — they are the "
            "alternate host. Apply myclobutanil or trifloxystrobin fungicide from pink bud "
            "stage through 3 weeks after petal fall. Resistant apple varieties are the "
            "best long-term solution."
        ),
        "sun": "Full sun. Open, breezy positions reduce spore deposit.",
        "water": "Avoid wetting leaves. Use drip irrigation or soaker hoses at ground level.",
        "nutrients": (
            "Balanced fertilization. Potassium and calcium help general disease resistance. "
            "Avoid over-fertilizing with nitrogen."
        ),
        "severity": "moderate",
    },

    "Apple___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Your apple plant looks healthy 🎉 Keep up your current care routine.",
        "sun": "Full sun, 6–8+ hours daily. Essential for fruit development and disease resistance.",
        "water": (
            "Deep watering 1–2×/week depending on rainfall. Allow top 2 inches of soil to "
            "dry between waterings. Mulch around base to retain moisture."
        ),
        "nutrients": (
            "NPK 10-10-10 in early spring. Side-dress with compost mid-season. "
            "Foliar micronutrient spray (zinc, boron, manganese) in summer supports fruit set."
        ),
        "severity": "none",
    },

    # ── BLUEBERRY ──────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Your blueberry plant is in great shape 🎉",
        "sun": "Full sun, 6–8 hrs/day for maximum berry production.",
        "water": (
            "Consistent moisture — blueberries are shallow-rooted. Water 1–2 inches/week. "
            "Mulch with pine bark or wood chips to retain moisture and keep pH low."
        ),
        "nutrients": (
            "Acidic soil fertilizer (pH 4.5–5.5). Use ammonium sulfate or acid-loving plant "
            "fertilizer. Avoid lime. Add sulfur to lower pH if needed."
        ),
        "severity": "none",
    },

    # ── CHERRY ─────────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "status": "diseased",
        "cure": (
            "Apply potassium bicarbonate, sulfur-based fungicide, or neem oil at first sign "
            "of white powdery coating. Repeat every 7–14 days. Remove and destroy heavily "
            "infected shoots. Improve air circulation by thinning branches. "
            "Avoid late-evening watering."
        ),
        "sun": "Full sun. Shaded, humid canopies strongly favor powdery mildew.",
        "water": "Water at soil level only. Never wet leaves. Morning watering if overhead is unavoidable.",
        "nutrients": (
            "Reduce nitrogen — lush new growth is most susceptible. "
            "Increase potassium and silica (potassium silicate spray) to harden leaf surfaces."
        ),
        "severity": "moderate",
    },

    "Cherry_(including_sour)___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Cherry plant looks healthy 🎉",
        "sun": "Full sun, 6–8 hrs/day. Critical for fruit sweetness and disease resistance.",
        "water": "Regular deep watering. Consistent moisture during fruit development prevents splitting.",
        "nutrients": "Balanced NPK in spring. Calcium spray during fruit development prevents tip burn.",
        "severity": "none",
    },

    # ── CORN (MAIZE) ───────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "status": "diseased",
        "cure": (
            "Apply foliar fungicides (azoxystrobin, pyraclostrobin, or propiconazole) at "
            "tasseling stage. Rotate crops — avoid planting corn in the same field 2 years "
            "in a row. Bury crop residue by tillage. Plant resistant hybrids in endemic areas."
        ),
        "sun": "Full sun. Poor light worsens the disease by slowing leaf drying.",
        "water": "Avoid overhead irrigation during humid periods. Ensure good field drainage.",
        "nutrients": (
            "Adequate nitrogen for canopy health, but avoid luxury levels. "
            "Potassium improves stalk strength and general disease tolerance."
        ),
        "severity": "moderate",
    },

    "Corn_(maize)___Common_rust_": {
        "status": "diseased",
        "cure": (
            "Apply triazole or strobilurin fungicides early — once rust is widespread, "
            "fungicides are less effective. Plant resistant hybrids. "
            "Scout fields from V6 stage onwards. Cool, humid weather (60–77°F) accelerates spread."
        ),
        "sun": "Full sun helps dry foliage and slow pustule formation.",
        "water": "Minimize leaf wetness duration. Drip or furrow irrigation preferred over overhead.",
        "nutrients": (
            "Balanced N-P-K. Adequate potassium improves overall stress tolerance. "
            "Sulfur applications have minor anti-fungal benefit."
        ),
        "severity": "moderate",
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "status": "diseased",
        "cure": (
            "Apply propiconazole, azoxystrobin, or mancozeb at tasseling if lesions appear "
            "before silking. Rotate with non-host crops (soybean, wheat). "
            "Bury corn residue. Choose resistant hybrids with Ht genes."
        ),
        "sun": "Full sun. High humidity and moderate temps (65–80°F) favor infection.",
        "water": "Avoid prolonged leaf wetness. Schedule irrigation for morning hours.",
        "nutrients": "Maintain adequate nitrogen — deficiency increases susceptibility. Side-dress if needed.",
        "severity": "severe",
    },

    "Corn_(maize)___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Corn looks healthy 🎉",
        "sun": "Full sun, 8+ hrs/day. Corn is a high-light crop.",
        "water": "1–1.5 inches/week. Critical during tasseling and ear fill — do not let it dry out.",
        "nutrients": (
            "Heavy feeder. Apply nitrogen in splits: at planting, V4–V6, and V10. "
            "Adequate phosphorus at planting for root development. Potassium for stalk strength."
        ),
        "severity": "none",
    },

    # ── GRAPE ──────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "status": "diseased",
        "cure": (
            "Apply myclobutanil or mancozeb fungicide from budbreak, repeating every 7–10 "
            "days through fruit set. Remove and destroy all mummified berries and infected "
            "canes. Improve canopy airflow by training and shoot positioning. "
            "Avoid wounds that let fungus enter."
        ),
        "sun": "Full sun is essential. Open canopy management is critical for disease control.",
        "water": "Drip irrigation only. Keep foliage dry at all times during the growing season.",
        "nutrients": (
            "Balanced vine nutrition. Avoid excess nitrogen post-fruit set. "
            "Adequate potassium supports berry skin development."
        ),
        "severity": "severe",
    },

    "Grape___Esca_(Black_Measles)": {
        "status": "diseased",
        "cure": (
            "No fully curative treatment currently available. Remove and destroy heavily "
            "infected vines. Apply sodium arsenite (where legally permitted) as trunk "
            "injection in dormancy — consult local regulations. Protect pruning wounds "
            "immediately with fungicidal paste (thiophanate-methyl). Minimize large pruning cuts."
        ),
        "sun": "Full sun. Stress from heat combined with Esca accelerates decline.",
        "water": (
            "Consistent irrigation to avoid water stress — drought dramatically worsens "
            "apoplexy (sudden collapse). Mulch to stabilize soil moisture."
        ),
        "nutrients": (
            "Balanced fertilization. Avoid over-cropping — stressed vines succumb faster. "
            "Boron and magnesium foliar sprays support vine health."
        ),
        "severity": "severe",
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "diseased",
        "cure": (
            "Apply copper-based fungicides or mancozeb at first symptom appearance. "
            "Remove infected leaves and destroy. Improve air circulation through canopy "
            "management. Avoid late-season nitrogen which promotes susceptible growth."
        ),
        "sun": "Full sun with good airflow around the canopy.",
        "water": "Water at the base. Avoid wetting foliage, especially late in the day.",
        "nutrients": "Balanced NPK. Reduce nitrogen in mid-to-late season.",
        "severity": "moderate",
    },

    "Grape___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Grapevine looks healthy 🎉",
        "sun": "Full sun, 7–8+ hrs/day. Essential for sugar development and disease resistance.",
        "water": "Deep, infrequent watering encourages deep rooting. Drip irrigation ideal.",
        "nutrients": (
            "Soil test annually. Potassium is critical for fruit quality. "
            "Moderate nitrogen. Magnesium deficiency (interveinal chlorosis) is common — apply Epsom salt if needed."
        ),
        "severity": "none",
    },

    # ── ORANGE ─────────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "status": "diseased",
        "cure": (
            "No cure currently exists. Infected trees should be removed and destroyed to "
            "prevent spread. Control the Asian citrus psyllid vector with systemic "
            "insecticides (imidacloprid soil drench). Use certified disease-free nursery stock. "
            "Research on thermotherapy and tetracycline injections is ongoing."
        ),
        "sun": "Full sun. Healthy growing conditions slow symptom progression slightly.",
        "water": "Consistent watering reduces tree stress. Mulch heavily to retain soil moisture.",
        "nutrients": (
            "Nutritional supplements (foliar zinc, manganese, iron) help maintain "
            "partial productivity in mildly affected trees. "
            "Potassium and magnesium support general tree health."
        ),
        "severity": "severe",
    },

    # ── PEACH ──────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "status": "diseased",
        "cure": (
            "Apply copper hydroxide or copper sulfate sprays from green tip through "
            "petal fall, and again in fall at leaf drop. Oxytetracycline sprays during "
            "bloom may help. Prune infected wood. Plant resistant varieties where available. "
            "Avoid overhead watering and working with wet trees."
        ),
        "sun": "Full sun. Well-ventilated, sunny positions reduce infection risk.",
        "water": "Drip irrigation. Avoid any leaf wetness. Bacterial spot spreads rapidly in wet conditions.",
        "nutrients": (
            "Avoid excess nitrogen — it promotes succulent tissue prone to infection. "
            "Adequate calcium and potassium strengthen cell walls."
        ),
        "severity": "moderate",
    },

    "Peach___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Peach plant looks healthy 🎉",
        "sun": "Full sun, 6–8 hrs/day minimum. Peaches require chilling hours and warm summers.",
        "water": "Deep watering weekly. Critical during fruit development. Mulch to retain moisture.",
        "nutrients": (
            "Nitrogen in early spring only — avoid late-season N which delays dormancy. "
            "Potassium for fruit quality. Zinc prevents little leaf disorder."
        ),
        "severity": "none",
    },

    # ── BELL PEPPER ────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "status": "diseased",
        "cure": (
            "Apply copper-based bactericides (copper hydroxide + mancozeb) every 5–7 days "
            "during wet weather. Remove heavily infected leaves. Avoid working in the field "
            "when plants are wet. Use disease-free certified seed. Crop rotation of 2–3 years."
        ),
        "sun": "Full sun. Good air movement around plants reduces disease spread.",
        "water": "Drip irrigation only. Keep foliage dry. Morning watering if overhead unavoidable.",
        "nutrients": (
            "Balanced NPK. Calcium spray reduces tissue susceptibility. "
            "Avoid excess nitrogen — it increases lesion size."
        ),
        "severity": "moderate",
    },

    "Pepper,_bell___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Bell pepper plant looks healthy 🎉",
        "sun": "Full sun, 6–8 hrs/day. Essential for fruit development and sweetness.",
        "water": "Consistent moisture — peppers are sensitive to drought and waterlogging both. 1–2 inches/week.",
        "nutrients": (
            "Phosphorus-rich fertilizer at transplant for root development. "
            "Switch to potassium-heavy fertilizer at flowering. Calcium prevents blossom end rot."
        ),
        "severity": "none",
    },

    # ── POTATO ─────────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "status": "diseased",
        "cure": (
            "Apply chlorothalonil, mancozeb, or azoxystrobin fungicide at first sign of "
            "target-like lesions. Repeat every 7–10 days in humid conditions. "
            "Remove lower infected leaves. Avoid overhead irrigation. Rotate crops."
        ),
        "sun": "Full sun. Dry, sunny conditions slow early blight progression.",
        "water": "Water at soil level. Allow foliage to dry completely. Avoid evening irrigation.",
        "nutrients": (
            "Ensure adequate nitrogen — nitrogen-stressed plants get early blight sooner. "
            "Potassium improves resistance. Balanced fertilization overall."
        ),
        "severity": "moderate",
    },

    "Potato___Late_blight": {
        "status": "diseased",
        "cure": (
            "Apply systemic fungicides: metalaxyl, cymoxanil, or dimethomorph at FIRST "
            "appearance — late blight spreads extremely fast. Destroy infected foliage "
            "immediately (burn or deep bury). Do NOT compost. Harvest tubers early if "
            "foliage is heavily infected. This disease destroyed the Irish Potato Famine — "
            "take it seriously and act fast."
        ),
        "sun": "Full sun and dry weather are the best natural suppressors.",
        "water": "Minimize leaf wetness at all costs. Late blight spreads explosively in wet, cool conditions.",
        "nutrients": (
            "Potassium is critical — it significantly improves late blight tolerance. "
            "Avoid excess nitrogen. Calcium strengthens cell walls against penetration."
        ),
        "severity": "severe",
    },

    "Potato___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Potato plant looks healthy 🎉",
        "sun": "Full sun, 6–8 hrs/day. Tuber formation needs warm days and cool nights.",
        "water": "Consistent moisture during tuber bulking — 1–2 inches/week. Irregular watering causes hollow heart.",
        "nutrients": (
            "High potassium requirement for tuber development. Moderate nitrogen. "
            "Sulfur improves flavor. Avoid fresh manure — use composted only."
        ),
        "severity": "none",
    },

    # ── RASPBERRY ──────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Raspberry plant looks healthy 🎉",
        "sun": "Full sun to partial shade. 6+ hrs preferred for best fruit production.",
        "water": "1–2 inches/week. Consistent moisture critical at fruiting. Excellent drainage essential.",
        "nutrients": (
            "Balanced NPK in early spring. Avoid nitrogen after midsummer — it delays hardening. "
            "Add compost annually. Slightly acidic soil (pH 5.5–6.5)."
        ),
        "severity": "none",
    },

    # ── SOYBEAN ────────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Soybean plant looks healthy 🎉",
        "sun": "Full sun, 6–8+ hrs/day.",
        "water": "1 inch/week. Critical at pod fill stage — drought stress at that point reduces yield significantly.",
        "nutrients": (
            "Soybeans fix their own nitrogen via rhizobia — inoculate seed if planting in a new field. "
            "Potassium and phosphorus are key yield drivers. Boron supports pod set."
        ),
        "severity": "none",
    },

    # ── SQUASH ─────────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "status": "diseased",
        "cure": (
            "Spray with potassium bicarbonate, neem oil, or diluted milk (40% milk + 60% water) "
            "weekly — milk has proven antifungal activity. Remove severely infected leaves. "
            "Sulfur-based fungicide for advanced infections. Improve air circulation around plants."
        ),
        "sun": "Full sun. Powdery mildew ironically thrives in dry, shaded conditions with high humidity.",
        "water": "Avoid wetting leaves. Morning watering at the base only.",
        "nutrients": (
            "Reduce nitrogen — excess N creates lush growth highly susceptible to mildew. "
            "Silica (potassium silicate spray) significantly hardens leaf surfaces against mildew."
        ),
        "severity": "moderate",
    },

    # ── STRAWBERRY ─────────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "status": "diseased",
        "cure": (
            "Apply myclobutanil or captan fungicide at first symptom. Remove infected leaves. "
            "Renovate beds after harvest by mowing and removing all debris. "
            "Plant resistant varieties. Ensure adequate spacing for airflow."
        ),
        "sun": "Full sun. Adequate spacing between plants reduces humidity buildup.",
        "water": "Drip irrigation. Avoid overhead watering which spreads fungal spores.",
        "nutrients": (
            "Balanced NPK. Avoid excess nitrogen post-harvest. "
            "Adequate potassium and calcium support berry and leaf health."
        ),
        "severity": "moderate",
    },

    "Strawberry___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Strawberry plant looks healthy 🎉",
        "sun": "Full sun, 6–8 hrs/day for sweetest berries.",
        "water": "1–1.5 inches/week. Consistent moisture at fruiting. Mulch with straw to retain moisture and keep fruit clean.",
        "nutrients": (
            "Balanced fertilizer at planting. Avoid high nitrogen after fruiting begins. "
            "Potassium boosts fruit sweetness and disease resistance."
        ),
        "severity": "none",
    },

    # ── TOMATO ─────────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "status": "diseased",
        "cure": (
            "Apply copper hydroxide + mancozeb tank mix every 5–7 days during wet weather. "
            "Remove infected lower leaves. Use certified disease-free transplants. "
            "Avoid working in the field when plants are wet. 2-year crop rotation."
        ),
        "sun": "Full sun. Good airflow reduces humidity around foliage.",
        "water": "Drip irrigation strictly. Bacterial spot spreads explosively through splash water.",
        "nutrients": "Balanced NPK. Calcium spray (calcium chloride) reduces susceptibility.",
        "severity": "moderate",
    },

    "Tomato___Early_blight": {
        "status": "diseased",
        "cure": (
            "Apply chlorothalonil or azoxystrobin fungicide every 7–10 days from first "
            "symptom. Remove lower infected leaves progressively. Mulch around base to "
            "prevent soil splash. Stake plants for good air circulation. Rotate crops annually."
        ),
        "sun": "Full sun. Remove shade-causing weeds or obstacles around plants.",
        "water": "Water at base only. Mulch heavily. Consistent moisture prevents stress which worsens blight.",
        "nutrients": (
            "Ensure adequate nitrogen — deficiency accelerates early blight. "
            "Potassium boosts overall disease resistance."
        ),
        "severity": "moderate",
    },

    "Tomato___Late_blight": {
        "status": "diseased",
        "cure": (
            "Act immediately — late blight can destroy a crop in days. Apply metalaxyl, "
            "cymoxanil, or chlorothalonil at first sign. Remove and bag infected plants — "
            "do not compost. Avoid overhead watering. In home gardens, copper fungicide "
            "is a good organic option. Cool, wet weather (50–70°F) is peak risk."
        ),
        "sun": "Full sun and warm dry days suppress late blight naturally.",
        "water": "Zero overhead watering. Drip only. Keep foliage completely dry.",
        "nutrients": (
            "Potassium dramatically improves late blight resistance — ensure it's adequate. "
            "Avoid luxury nitrogen. Calcium strengthens cell walls."
        ),
        "severity": "severe",
    },

    "Tomato___Leaf_Mold": {
        "status": "diseased",
        "cure": (
            "Apply copper fungicide, chlorothalonil, or mancozeb. Remove badly infected leaves. "
            "Reduce humidity: ventilate greenhouses, increase plant spacing. "
            "Leaf mold is primarily a greenhouse disease — humidity above 85% drives infection."
        ),
        "sun": "Full sun and ventilation are the primary preventive measures.",
        "water": "No overhead watering. Reduce humidity around the canopy at all costs.",
        "nutrients": "Balanced nutrition. Avoid excess nitrogen that creates dense, humid canopies.",
        "severity": "moderate",
    },

    "Tomato___Septoria_leaf_spot": {
        "status": "diseased",
        "cure": (
            "Apply chlorothalonil or copper fungicide at first symptom. Remove lower infected "
            "leaves and destroy (do not compost). Mulch heavily to prevent soil splash. "
            "Stake and cage plants for airflow. Avoid working with wet plants. Annual crop rotation."
        ),
        "sun": "Full sun with good airflow. Dense canopies trap humidity and worsen Septoria.",
        "water": "Drip irrigation at base. Mulch to prevent soil splash which spreads spores.",
        "nutrients": "Adequate nitrogen prevents stress-driven susceptibility. Balanced overall.",
        "severity": "moderate",
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "status": "diseased",
        "cure": (
            "Spray forcefully with water to dislodge mites from undersides of leaves. "
            "Apply neem oil, insecticidal soap, or miticide (abamectin, bifenazate) — "
            "rotate chemicals to prevent resistance. Introduce predatory mites (Phytoseiulus persimilis) "
            "for biological control. Spider mites thrive in hot, dry, dusty conditions."
        ),
        "sun": "Full sun is fine — spider mites prefer hot, dry conditions, so increase humidity slightly.",
        "water": (
            "Increase watering slightly and mist the air around plants — "
            "spider mites hate humidity. But don't wet foliage excessively."
        ),
        "nutrients": (
            "Well-nourished plants tolerate mite pressure better. Avoid excess nitrogen "
            "which produces preferred soft leaf tissue for mites."
        ),
        "severity": "moderate",
    },

    "Tomato___Target_Spot": {
        "status": "diseased",
        "cure": (
            "Apply azoxystrobin, chlorothalonil, or mancozeb fungicide. Remove and destroy "
            "infected leaves. Improve canopy airflow by pruning suckers. Avoid overhead "
            "irrigation. Crop rotation helps reduce soilborne inoculum."
        ),
        "sun": "Full sun and good air circulation are the first line of defense.",
        "water": "Drip irrigation. Do not wet foliage during the growing season.",
        "nutrients": "Balanced NPK. Potassium improves general fungal resistance.",
        "severity": "moderate",
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "status": "diseased",
        "cure": (
            "No cure — TYLCV is a viral disease spread by whiteflies. Remove and destroy "
            "infected plants immediately to limit spread. Control whitefly vector aggressively: "
            "yellow sticky traps, imidacloprid soil drench, or spinosad. Use reflective mulch "
            "to repel whiteflies. Plant resistant varieties (many modern hybrids carry Ty genes)."
        ),
        "sun": "Full sun. Healthy growing conditions don't cure the virus but support plant vigor.",
        "water": "Regular consistent watering to maintain plant vigor and reduce stress.",
        "nutrients": (
            "Avoid excess nitrogen — it attracts whiteflies. "
            "Potassium and calcium support general plant resilience."
        ),
        "severity": "severe",
    },

    "Tomato___Tomato_mosaic_virus": {
        "status": "diseased",
        "cure": (
            "No cure. Remove and destroy infected plants. Disinfect all tools with 10% bleach "
            "or 70% alcohol between uses — ToMV is extremely persistent on surfaces. "
            "Wash hands before handling plants. Avoid tobacco products near plants (ToMV is "
            "related to Tobacco Mosaic Virus). Use resistant varieties (Tm-2 gene)."
        ),
        "sun": "Full sun supports vigor in mildly infected plants.",
        "water": "Regular watering. Avoid stressing the plant — stress worsens viral symptoms.",
        "nutrients": "Well-balanced nutrition. Phosphorus and potassium support general resilience.",
        "severity": "severe",
    },

    "Tomato___healthy": {
        "status": "healthy",
        "cure": "No disease detected! Tomato plant looks healthy 🎉",
        "sun": "Full sun, 8+ hrs/day. Tomatoes are one of the most sun-hungry crops.",
        "water": (
            "Deep, consistent watering — 1–2 inches/week. Irregular watering causes "
            "blossom end rot and fruit cracking. Mulch heavily. "
            "Water at base only to prevent foliar diseases."
        ),
        "nutrients": (
            "Heavy feeder. High nitrogen at transplant, then switch to low-N, high-P/K "
            "fertilizer at flowering. Calcium prevents blossom end rot. "
            "Magnesium (Epsom salt foliar spray) prevents inter-veinal yellowing."
        ),
        "severity": "none",
    },
}


def get_disease_info(label: str) -> dict:
    """
    Returns disease info for a given label string.
    Falls back to a generic entry if label not found.
    """
    return DISEASE_INFO.get(label, {
        "status": "unknown",
        "cure": "Information not available for this class. Please consult a local agricultural extension service.",
        "sun": "Provide full sun (6–8 hrs/day) as a general guideline for most crops.",
        "water": "Water consistently at the base, 1–2 inches per week.",
        "nutrients": "Use a balanced NPK fertilizer and amend soil with compost annually.",
        "severity": "unknown",
    })
