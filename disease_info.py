DISEASE_GUIDE = [
    {
        "crop": "Apple", "disease": "Apple scab", "symptoms": "Olive-green to dark, velvety spots on leaves or fruit; leaves may curl or drop early.",
        "causes": "A fungal disease favored by wet spring weather and infected leaf litter.",
        "prevention": "Remove fallen leaves, improve air movement, and choose locally suitable resistant cultivars where available.",
        "management": "Track symptoms over time and ask a local extension service about region- and cultivar-specific options.",
    },
    {
        "crop": "Corn", "disease": "Common rust", "symptoms": "Raised, reddish-brown pustules scattered across leaf surfaces.",
        "causes": "A wind-dispersed fungal disease that can spread during mild, humid conditions.",
        "prevention": "Use locally adapted varieties and monitor fields during periods of humid weather.",
        "management": "Compare symptoms with an extension guide; consider crop stage and local disease pressure before acting.",
    },
    {
        "crop": "Grape", "disease": "Black rot", "symptoms": "Tan leaf lesions with dark margins; fruit may shrivel into dark, firm mummies.",
        "causes": "A fungal disease that survives in infected fruit and plant debris.",
        "prevention": "Remove mummified berries and infected debris; maintain canopy airflow.",
        "management": "Seek local viticulture guidance to confirm the diagnosis and decide appropriate next steps.",
    },
    {
        "crop": "Potato", "disease": "Early blight", "symptoms": "Older leaves may develop brown target-like spots with concentric rings and yellowing.",
        "causes": "A fungal disease often associated with plant stress, warm conditions, and leaf wetness.",
        "prevention": "Rotate crops, reduce prolonged leaf wetness, and maintain balanced plant nutrition.",
        "management": "Remove heavily affected debris where practical and consult local agricultural guidance.",
    },
    {
        "crop": "Tomato", "disease": "Early blight", "symptoms": "Dark leaf spots with concentric rings, commonly beginning on older leaves.",
        "causes": "A fungal disease favored by warm, wet conditions and surviving crop debris.",
        "prevention": "Rotate planting areas, water near the soil, mulch to limit soil splash, and improve airflow.",
        "management": "Remove affected leaves when practical and confirm with a local extension specialist.",
    },
    {
        "crop": "Tomato", "disease": "Late blight", "symptoms": "Fast-spreading water-soaked lesions that may develop pale growth in humid conditions.",
        "causes": "A destructive oomycete disease that spreads rapidly in cool, wet weather.",
        "prevention": "Use healthy transplants, avoid overhead watering, and monitor local disease alerts.",
        "management": "Isolate suspicious plants and contact an agricultural extension service promptly for confirmation.",
    },
    {
        "crop": "Healthy", "disease": "Healthy leaf", "symptoms": "No clear disease-like pattern is visible in the submitted image.",
        "causes": "A photo-based model result is not proof that a plant is disease-free.",
        "prevention": "Continue routine scouting and take clear photos of any new or changing symptoms.",
        "management": "Seek expert advice if the plant shows symptoms despite a healthy model result.",
    },
]


def guide_matches(query: str = ""):
    normalized = query.strip().casefold()
    if not normalized:
        return DISEASE_GUIDE
    return [
        entry for entry in DISEASE_GUIDE
        if normalized in " ".join(str(value) for value in entry.values()).casefold()
    ]


def find_guidance(crop_name: str, disease_name: str):
    crop = crop_name.casefold()
    disease = disease_name.casefold()
    for entry in DISEASE_GUIDE:
        if entry["crop"].casefold() in crop and entry["disease"].casefold() in disease:
            return entry
    if "healthy" in disease:
        return next(entry for entry in DISEASE_GUIDE if entry["crop"] == "Healthy")
    return None