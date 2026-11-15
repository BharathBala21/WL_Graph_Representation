"""Display names for OLID-I crops and condition codes.

Same vocabulary as Application/scripts/build_plantcare_data.py (corrected OLID-I codes).
"""

CROPS = {
    'ash_gourd': 'Ash gourd', 'bitter_gourd': 'Bitter gourd', 'bottle_gourd': 'Bottle gourd',
    'cucumber': 'Cucumber', 'eggplant': 'Eggplant', 'ridge_gourd': 'Ridge gourd',
    'snake_gourd': 'Snake gourd', 'tomato': 'Tomato',
}

# code -> {name, type}
CONDITIONS = {
    'healthy': {'name': 'Healthy', 'type': 'Control'},
    'N': {'name': 'Nitrogen deficiency', 'type': 'Nutrient deficiency'},
    'K': {'name': 'Potassium deficiency', 'type': 'Nutrient deficiency'},
    'N_K': {'name': 'Nitrogen + potassium deficiency', 'type': 'Nutrient deficiency'},
    'N_Mg': {'name': 'Nitrogen + magnesium deficiency', 'type': 'Nutrient deficiency'},
    'K_Mg': {'name': 'Potassium + magnesium deficiency', 'type': 'Nutrient deficiency'},
    'MIT': {'name': 'Mite', 'type': 'Pest'},
    'JAS': {'name': 'Jassid', 'type': 'Pest'},
    'JAS_MIT': {'name': 'Jassid + mite', 'type': 'Pest'},
    'EB': {'name': 'Epilachna beetle', 'type': 'Pest'},
    'MIT_EB': {'name': 'Mite + Epilachna beetle', 'type': 'Pest'},
    'FB': {'name': 'Flea beetle', 'type': 'Pest'},
    'LM': {'name': 'Leaf miner', 'type': 'Pest'},
    'PLEI': {'name': 'Pumpkin leaf-eating insect', 'type': 'Pest'},
    'PLEI_IEM': {'name': 'Pumpkin leaf-eating insect + egg mass', 'type': 'Pest'},
    'PLEI_MIT': {'name': 'Pumpkin leaf-eating insect + mite', 'type': 'Pest'},
    'PC': {'name': 'Pumpkin caterpillar', 'type': 'Pest'},
    'PM': {'name': 'Powdery mildew', 'type': 'Disease'},
    'DM': {'name': 'Downy mildew', 'type': 'Disease'},
    'LS': {'name': 'Leaf spot', 'type': 'Disease'},
}

# Plain-language names for the 38 measurements (used to explain results)
FEATURE_NAMES = {
    'area': 'Leaf area', 'perimeter': 'Leaf perimeter', 'aspect_ratio': 'Length-to-width ratio',
    'circularity': 'Roundness', 'extent': 'Fill of bounding box', 'solidity': 'Solidity (no gaps or cuts)',
    'convex_area': 'Outline area', 'major_axis_length': 'Leaf length', 'minor_axis_length': 'Leaf width',
    'eccentricity': 'Elongation',
    'mean_R': 'Average red', 'std_R': 'Red variation', 'mean_G': 'Average green', 'std_G': 'Green variation',
    'mean_B': 'Average blue', 'std_B': 'Blue variation',
    'mean_H': 'Average hue', 'std_H': 'Hue variation', 'mean_S': 'Colour saturation', 'std_S': 'Saturation variation',
    'mean_V': 'Brightness', 'std_V': 'Brightness variation',
    'mean_L': 'Lightness', 'std_L': 'Lightness variation', 'mean_a': 'Green-red balance', 'std_a': 'Green-red variation',
    'mean_b': 'Blue-yellow balance', 'std_b': 'Blue-yellow variation',
    'green_ratio': 'Greenness', 'yellow_ratio': 'Yellowness',
    'glcm_contrast': 'Texture contrast', 'glcm_homogeneity': 'Texture smoothness', 'glcm_energy': 'Texture uniformity',
    'glcm_correlation': 'Texture regularity',
    'core_mean_H': 'Hue at leaf centre', 'core_yellow_ratio': 'Yellowness at leaf centre',
    'edge_mean_H': 'Hue at leaf edge', 'edge_yellow_ratio': 'Yellowness at leaf edge',
}
