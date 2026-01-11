# src/config.py

# Theme Colors
COLOR_NAVY = "#0A1628"
COLOR_GOLD = "#D4AF37"
COLOR_CHARCOAL = "#1E1E1E"
COLOR_AMBER = "#FFBF00"  # Evasive
COLOR_WHITE = "#FFFFFF"
COLOR_GREY = "#7F8C8D"
COLOR_GREEN = "#2ECC71"
COLOR_RED = "#E74C3C"

# Stance Color Mapping
STANCE_COLORS = {
    "Pro": COLOR_GREEN,
    "Con": COLOR_RED,
    "Neutral": COLOR_GREY,
    "Evasive": COLOR_AMBER
}

# Topic ID to Human-Friendly Labels Mapping
TOPIC_MAP = {
    -1: "General Noise",
    0: "Political Banter",
    1: "Ministerial Procedures",
    2: "Honorifics & Greetings",
    3: "National Administration",
    4: "Procedural Interjections",
    5: "Education & Schools",
    6: "Committee Business",
    7: "Greetings & Acknowledgments",
    8: "Procedural Interjections",
    9: "General Negations",
    10: "Budget & Finance",
    11: "Time Management",
    12: "General Discussions",
    13: "Urban & Town Planning",
    14: "Islamic Affairs & Syariah",
    15: "Healthcare & Hospitals",
    16: "Regional (KL/Selayang)",
    17: "Procedural (Standing)",
    20: "Water & Infrastructure",
    21: "Regional (Kota Tinggi)",
    22: "Housing & Development",
    24: "Taxation & GST",
    26: "Regional (Kelantan)",
    27: "Regional (Cameron Highlands)",
    30: "Entrepreneurship & SMEs",
    33: "Telecommunications & Digital",
    34: "Public Health & Tobacco",
    37: "Oil & Gas Industry",
    38: "Women & Gender Policy",
    41: "1MDB & Asset Recovery",
    44: "Public Safety & Police",
    47: "Regional (Hulu Langat)",
    48: "Financial Matters"
}

# Topics to hide by default (Noise, Procedural, Greetings, etc.)
NOISE_TOPICS = [-1, 0, 1, 2, 4, 7, 8, 9, 11, 12, 17, 18, 19, 21, 23, 25, 28, 29, 31, 32, 35, 36, 39, 40, 42, 43, 45, 46]

# Layout Constants
APP_TITLE = "Parliamentary Intelligence Dashboard"
APP_SUBTITLE = "Malaysian Hansard Analytics & Evasiveness Detector"
UNIVERSITY_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/University_of_Malaya_coat_of_arms.svg/300px-University_of_Malaya_coat_of_arms.svg.png"
