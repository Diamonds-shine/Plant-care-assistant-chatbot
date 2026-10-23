"""Small, transparent synonym and spelling maps.

These are applied to the *normalized* text before tokenization, so both the
knowledge-base documents and the user's question are rewritten the same way.
They complement TF-IDF (they do not replace it): TF-IDF only matches words
that are literally shared, so mapping "money plant" -> "pothos" lets the two
names land on the same vector dimensions.

Keep entries lower-case and written the way `normalize()` outputs text
(apostrophes removed, hyphens turned into spaces).
"""

# Multi-word phrases first. Longer phrases are always applied before shorter
# ones (see `apply_phrase_synonyms` in preprocess.py), so "chinese money plant"
# is handled before "money plant".
PHRASE_SYNONYMS: dict[str, str] = {
    # Plant name aliases (common Indian / regional names included)
    "chinese money plant": "pilea",
    "pancake plant": "pilea",
    "ufo plant": "pilea",
    "money plant": "pothos",
    "money plants": "pothos",
    "devils ivy": "pothos",
    "devil ivy": "pothos",
    "mother in laws tongue": "snake plant",
    "mother in law tongue": "snake plant",
    "mother in law s tongue": "snake plant",
    "sansevieria": "snake plant",
    "sansevaria": "snake plant",
    "zanzibar gem": "zz plant",
    "zamioculcas": "zz plant",
    "zz plants": "zz plant",
    "rubber plant": "rubber tree",
    "parlour palm": "parlor palm",
    "donkeys tail": "burros tail",
    "donkey tail": "burros tail",
    "holiday cactus": "christmas cactus",
    "house plants": "indoor plants",
    "house plant": "indoor plant",
    "houseplants": "indoor plants",
    "houseplant": "indoor plant",
    # Pests and symptoms described in everyday words
    "tiny flies": "fungus gnats",
    "small flies": "fungus gnats",
    "little flies": "fungus gnats",
    "little black flies": "fungus gnats",
    "small black flies": "fungus gnats",
    "fungus gnats": "fungus gnats",  # identity entry so "gnats" below does not double it
    "fungus gnat": "fungus gnats",
    "gnats": "fungus gnats",
    "plant food": "fertilizer",
    "white fluffy": "mealybugs",
    "cotton like": "mealybugs",
    # Spelling variants (British -> American spelling used in the knowledge base)
    "fertiliser": "fertilizer",
    "fertilisers": "fertilizers",
    "fertilise": "fertilize",
    "fertilising": "fertilizing",
    "colour": "color",
    # Common misspellings seen in student testing
    "watring": "watering",
    "waterring": "watering",
    "fertlizer": "fertilizer",
    "fertilzer": "fertilizer",
    "sunlite": "sunlight",
    "sun light": "sunlight",
    "succulant": "succulent",
    "succulants": "succulents",
    "repoting": "repotting",
    "leafs": "leaves",
    "yelow": "yellow",
    "yello": "yellow",
    "overwatring": "overwatering",
    "over watering": "overwatering",
    "over watered": "overwatered",
    "under watering": "underwatering",
    "under watered": "underwatered",
}

# Single words mapped to a shared concept word.
WORD_SYNONYMS: dict[str, str] = {
    "bugs": "pests",
    "bug": "pest",
    "insects": "pests",
    "insect": "pest",
    "cat": "pet",
    "cats": "pets",
    "kitten": "pet",
    "kittens": "pets",
    "dog": "pet",
    "dogs": "pets",
    "puppy": "pet",
    "puppies": "pets",
    "poisonous": "toxic",
    "poison": "toxic",
    "harmful": "toxic",
    "dangerous": "toxic",
}

# Display aliases used by the dataset builder to write extra alternative
# questions for plant entries (e.g. "How often should I water a money plant?").
# Keys are FloraDB `common_name` values.
PLANT_ALIASES: dict[str, list[str]] = {
    "Swiss Cheese Plant": ["monstera", "monstera deliciosa"],
    "Golden Pothos": ["pothos", "money plant", "devil's ivy"],
    "Snake Plant": ["sansevieria", "mother-in-law's tongue"],
    "ZZ Plant": ["zanzibar gem", "zamioculcas"],
    "Peace Lily": ["spathiphyllum"],
    "Flamingo Lily": ["anthurium"],
    "Rubber Tree": ["rubber plant", "ficus elastica"],
    "Fiddle Leaf Fig": ["fiddle leaf", "ficus lyrata"],
    "Weeping Fig": ["ficus benjamina", "ficus tree"],
    "Jade Plant": ["jade", "crassula"],
    "Aloe Vera": ["aloe"],
    "Spider Plant": ["airplane plant"],
    "Chinese Money Plant": ["pilea", "pancake plant"],
    "Money Tree": ["pachira"],
    "Christmas Cactus": ["holiday cactus"],
    "Boston Fern": ["sword fern"],
    "Parlor Palm": ["parlour palm"],
    "Arrowhead Vine": ["syngonium", "arrowhead plant"],
    "Dumb Cane": ["dieffenbachia"],
    "Corn Plant": ["dracaena"],
    "Wax Plant": ["hoya"],
    "String of Hearts": ["rosary vine"],
    "Nerve Plant": ["fittonia"],
    "Prayer Plant": ["maranta"],
    "Wandering Dude": ["inch plant", "tradescantia"],
    "African Violet": ["saintpaulia"],
    "Chinese Evergreen": ["aglaonema"],
    "Burro's Tail": ["donkey's tail"],
    "Polka Dot Begonia": ["begonia maculata"],
    "Florist's Cyclamen": ["cyclamen"],
    "Chinese Hibiscus": ["hibiscus"],
    "Paper Flower": ["bougainvillea"],
    "Mexican Snowball": ["echeveria"],
    "Old Lady Cactus": ["mammillaria"],
    "Living Stones": ["lithops"],
    "Scarlet Star Bromeliad": ["bromeliad", "guzmania"],
    "Lucky Bamboo": ["dracaena sanderiana"],
}
