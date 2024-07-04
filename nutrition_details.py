import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import os

# Load pre-trained EfficientNetB0 model
model = EfficientNetB0(weights='imagenet')

# Mapping specific fruit varieties to general types
variety_to_general = {
    'Granny_Smith': 'apple',
    'Red_Delicious': 'apple',
    'Gala': 'apple',
    'Fuji': 'apple',
    'Golden_Delicious': 'apple',
    'Honeycrisp': 'apple',
    'McIntosh': 'apple',
    'Braeburn': 'apple',
    'Cortland': 'apple',
    'Empire': 'apple',
    'Cavendish': 'banana',
    'Orin': 'banana',
    'Dole': 'banana',
    'Lady_Finger': 'banana',
    'Plantain': 'banana',
    'Red_Banana': 'banana',
    'Burro_Banana': 'banana',
    'Manzano': 'banana',
    'Navel': 'orange',
    'Valencia': 'orange',
    'Blood_Orange': 'orange',
    'Mandarin': 'orange',
    'Tangerine': 'orange',
    'Clementine': 'orange',
    'Satsuma': 'orange',
    'Bell_Pepper': 'pepper',
    'Red_Pepper': 'pepper',
    'Green_Pepper': 'pepper',
    'Yellow_Pepper': 'pepper',
    'Orange_Pepper': 'pepper',
    'Chili_Pepper': 'pepper',
    'Jalapeno': 'pepper',
    'Poblano': 'pepper',
    'Habanero': 'pepper',
    'Serrano': 'pepper',
    'Anaheim': 'pepper',
    'Scotch_Bonnet': 'pepper',
    'Cucumber': 'cucumber',
    'Gherkin': 'cucumber',
    'Persian_Cucumber': 'cucumber',
    'Zucchini': 'zucchini',
    'Yellow_Zucchini': 'zucchini',
    'Tomato': 'tomato',
    'Cherry_Tomato': 'tomato',
    'Beefsteak_Tomato': 'tomato',
    'Roma_Tomato': 'tomato',
    'Grape_Tomato': 'tomato',
    'Heirloom_Tomato': 'tomato',
    'Carrot': 'carrot',
    'Baby_Carrot': 'carrot',
    'Rainbow_Carrot': 'carrot',
    'Potato': 'potato',
    'Sweet_Potato': 'potato',
    'Yam': 'potato',
    'Russet_Potato': 'potato',
    'Red_Potato': 'potato',
    'Fingerling_Potato': 'potato',
    'Eggplant': 'eggplant',
    'Japanese_Eggplant': 'eggplant',
    'Chinese_Eggplant': 'eggplant',
    'Italian_Eggplant': 'eggplant',
    'Thai_Eggplant': 'eggplant',
    'Onion': 'onion',
    'Red_Onion': 'onion',
    'Yellow_Onion': 'onion',
    'White_Onion': 'onion',
    'Green_Onion': 'onion',
    'Shallot': 'onion',
    'Leek': 'onion',
    'Watermelon': 'watermelon',
    'Cantaloupe': 'melon',
    'Honeydew': 'melon',
    'Canary_Melon': 'melon',
    'Galia_Melon': 'melon',
    'Pumpkin': 'pumpkin',
    'Butternut_Squash': 'squash',
    'Acorn_Squash': 'squash',
    'Spaghetti_Squash': 'squash',
    'Kabocha_Squash': 'squash',
    'Delicata_Squash': 'squash',
    'Buttercup_Squash': 'squash',
    'Hubbard_Squash': 'squash',
    'Avocado': 'avocado',
    'Papaya': 'papaya',
    'Mango': 'mango',
    'Pineapple': 'pineapple',
    'Kiwi': 'kiwi',
    'Peach': 'peach',
    'Plum': 'plum',
    'Pear': 'pear',
    'Nectarine': 'nectarine',
    'Grapefruit': 'grapefruit',
    'Lemon': 'lemon',
    'Lime': 'lime',
    'Ginger': 'ginger',
    'Garlic': 'garlic',
    'Radish': 'radish',
    'Turnip': 'turnip',
    'Parsnip': 'parsnip',
    'Beet': 'beet',
    'Celery': 'celery',
    'Fennel': 'fennel',
    'Artichoke': 'artichoke',
    'Broccoli': 'broccoli',
    'Cauliflower': 'cauliflower',
    'Brussels_Sprout': 'brussels_sprout',
    'Cabbage': 'cabbage',
    'Kale': 'kale',
    'Spinach': 'spinach',
    'Swiss_Chard': 'chard',
    'Collard_Greens': 'collard_greens',
    'Mustard_Greens': 'mustard_greens',
    'Bok_Choy': 'bok_choy',
    'Radicchio': 'radicchio',
    'Endive': 'endive',
    'Arugula': 'arugula',
    'Dandelion_Greens': 'dandelion_greens',
    'Lettuce': 'lettuce',
    'Romaine': 'lettuce',
    'Butterhead': 'lettuce',
    'Iceberg': 'lettuce',
    'Oakleaf': 'lettuce',
    'Frisee': 'lettuce',
    'Batavia': 'lettuce',
    'Mizuna': 'mizuna',
    'Watercress': 'watercress',
    'Sorrel': 'sorrel',
    'Chicory': 'chicory',
    'Escarole': 'escarole',
    'Purslane': 'purslane',
    'Tat_Soi': 'tat_soi',
    'Mache': 'mache',
    'Parsley': 'parsley',
    'Basil': 'basil',
    'Cilantro': 'cilantro',
    'Mint': 'mint',
    'Oregano': 'oregano',
    'Thyme': 'thyme',
    'Rosemary': 'rosemary',
    'Sage': 'sage',
    'Tarragon': 'tarragon',
    'Chervil': 'chervil',
    'Dill': 'dill',
    'Chive': 'chive',
    'Marjoram': 'marjoram',
    'Cress': 'cress',
    'Lovage': 'lovage',
    'Anise': 'anise',
    'Bay_Leaf': 'bay_leaf',
    'Caraway': 'caraway',
    'Cardamom': 'cardamom',
    'Cinnamon': 'cinnamon',
    'Clove': 'clove',
    'Cumin': 'cumin',
    'Coriander': 'coriander',
    'Fennel_Seed': 'fennel_seed',
    'Fenugreek': 'fenugreek',
    'Nutmeg': 'nutmeg',
    'Paprika': 'paprika',
    'Peppercorn': 'peppercorn',
    'Poppy_Seed': 'poppy_seed',
    'Saffron': 'saffron',
    'Sesame_Seed': 'sesame_seed',
    'Turmeric': 'turmeric',
    'Vanilla': 'vanilla'
}


# Function to preprocess image for recognition
def preprocess_image(img):
    img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Function to recognize food
def recognize_food(img):
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1], decoded_predictions[0][0][2]  # (label, confidence)


# Function to resize and pad image to a target size
def resize_and_pad(img, target_size=(700, 700)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w, pad_h = (target_size[1] - new_w) // 2, (target_size[0] - new_h) // 2
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    return padded_img, scale


# Function to estimate ellipsoid volume
def estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio=1.2):
    major_radius_mm = np.sqrt(area_pixels / np.pi) * pixel_to_mm_ratio
    minor_radius_mm = major_radius_mm / height_to_width_ratio
    volume_mm3 = (4 / 3) * np.pi * (major_radius_mm / 2) * (minor_radius_mm / 2) ** 2
    return volume_mm3


# Function to estimate cylinder volume
def estimate_cylinder_volume(area_pixels, pixel_to_mm_ratio, height_to_diameter_ratio=1):
    radius_mm = np.sqrt(area_pixels / np.pi) * pixel_to_mm_ratio
    height_mm = radius_mm * height_to_diameter_ratio
    volume_mm3 = np.pi * (radius_mm ** 2) * height_mm
    return volume_mm3


# Function to estimate cone volume
def estimate_cone_volume(area_pixels, pixel_to_mm_ratio, height_to_base_diameter_ratio=1):
    base_radius_mm = np.sqrt(area_pixels / np.pi) * pixel_to_mm_ratio
    height_mm = base_radius_mm * height_to_base_diameter_ratio
    volume_mm3 = (1 / 3) * np.pi * (base_radius_mm ** 2) * height_mm
    return volume_mm3


# Function to estimate volume based on shape
def estimate_volume_based_on_shape(label, area_pixels, pixel_to_mm_ratio):
    general_label = variety_to_general.get(label, label).lower()

    if 'apple' in general_label:
        height_to_width_ratio = 1.2  # Typical for apples
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'banana' in general_label:
        height_to_diameter_ratio = 10
        volume_mm3 = estimate_cylinder_volume(area_pixels, pixel_to_mm_ratio, height_to_diameter_ratio)
    elif 'orange' in general_label or 'grapefruit' in general_label or 'lemon' in general_label or 'lime' in general_label:
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio=1)
    elif 'cucumber' in general_label or 'zucchini' in general_label:
        height_to_diameter_ratio = 8
        volume_mm3 = estimate_cylinder_volume(area_pixels, pixel_to_mm_ratio, height_to_diameter_ratio)
    elif 'pepper' in general_label:
        height_to_width_ratio = 1.5
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'tomato' in general_label or 'kiwi' in general_label:
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio=1)
    elif 'carrot' in general_label or 'parsnip' in general_label or 'turnip' in general_label:
        height_to_base_diameter_ratio = 10
        volume_mm3 = estimate_cone_volume(area_pixels, pixel_to_mm_ratio, height_to_base_diameter_ratio)
    elif 'potato' in general_label or 'yam' in general_label or 'sweet_potato' in general_label:
        height_to_width_ratio = 1.3
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'eggplant' in general_label:
        height_to_width_ratio = 3
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'onion' in general_label or 'garlic' in general_label:
        height_to_width_ratio = 1.2
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'watermelon' in general_label or 'melon' in general_label:
        height_to_width_ratio = 1.5
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'pumpkin' in general_label or 'squash' in general_label:
        height_to_width_ratio = 1.2
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'pineapple' in general_label:
        height_to_diameter_ratio = 5
        volume_mm3 = estimate_cylinder_volume(area_pixels, pixel_to_mm_ratio, height_to_diameter_ratio)
    elif 'pear' in general_label or 'avocado' in general_label:
        height_to_width_ratio = 1.5
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'peach' in general_label or 'plum' in general_label or 'nectarine' in general_label:
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio=1)
    elif 'mango' in general_label or 'papaya' in general_label:
        height_to_width_ratio = 1.8
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'cabbage' in general_label or 'lettuce' in general_label or 'kale' in general_label:
        height_to_width_ratio = 1
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif 'broccoli' in general_label or 'cauliflower' in general_label or 'brussels_sprout' in general_label:
        height_to_width_ratio = 1.5
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio, height_to_width_ratio)
    elif general_label in [
        'anise', 'artichoke', 'arugula', 'basil', 'bay_leaf', 'beet', 'bok_choy', 'caraway', 'cardamom', 'celery',
        'chard', 'chervil', 'chicory', 'chive', 'cilantro', 'cinnamon', 'clove', 'collard_greens', 'coriander',
        'cress', 'cumin', 'dandelion_greens', 'dill', 'endive', 'escarole', 'fennel', 'fennel_seed', 'fenugreek',
        'ginger', 'lovage', 'mache', 'marjoram', 'mint', 'mizuna', 'mustard_greens', 'nutmeg', 'oregano', 'paprika',
        'parsley', 'peppercorn', 'poppy_seed', 'purslane', 'radicchio', 'radish', 'rosemary', 'saffron', 'sage',
        'sesame_seed', 'sorrel', 'spinach', 'tarragon', 'tat_soi', 'thyme', 'turmeric', 'vanilla', 'watercress'
    ]:
        # Assuming irregular shape for herbs and leafy greens
        volume_mm3 = area_pixels * pixel_to_mm_ratio

    else:
        # Default case, assuming ellipsoid if the general shape is not explicitly known
        volume_mm3 = estimate_ellipsoid_volume(area_pixels, pixel_to_mm_ratio)

    return volume_mm3


# Function to estimate weight based on volume
def estimate_weight(volume_cm3, fruit_type):
    # Weight estimation factors (grams per cm³)
    weight_factors = {
        'anise': 0.45, 'apple': 0.5, 'artichoke': 0.8, 'arugula': 0.35, 'avocado': 0.9, 'banana': 0.8, 'basil': 0.25,
        'bay_leaf': 0.3, 'beet': 0.75, 'bok_choy': 0.55, 'broccoli': 0.6, 'brussels_sprout': 0.7, 'cabbage': 0.65,
        'caraway': 0.4, 'cardamom': 0.3, 'carrot': 0.85, 'cauliflower': 0.55, 'celery': 0.35, 'chard': 0.6,
        'chervil': 0.2, 'chicory': 0.45, 'chive': 0.25, 'cilantro': 0.25, 'cinnamon': 0.2, 'clove': 0.3,
        'collard_greens': 0.55, 'coriander': 0.25, 'cress': 0.3, 'cucumber': 0.6, 'cumin': 0.4, 'dandelion_greens': 0.5,
        'dill': 0.25, 'eggplant': 0.75, 'endive': 0.35, 'escarole': 0.45, 'fennel': 0.4, 'fennel_seed': 0.2,
        'fenugreek': 0.3, 'garlic': 0.4, 'ginger': 0.45, 'grapefruit': 0.8, 'kale': 0.5, 'kiwi': 0.75, 'lemon': 0.7,
        'lettuce': 0.35, 'lime': 0.65, 'lovage': 0.3, 'mache': 0.35, 'mango': 0.7, 'marjoram': 0.2, 'melon': 0.9,
        'mint': 0.25, 'mizuna': 0.35, 'mustard_greens': 0.4, 'nectarine': 0.65, 'nutmeg': 0.3, 'onion': 0.75,
        'orange': 0.7, 'oregano': 0.25, 'papaya': 0.75, 'paprika': 0.25, 'parsley': 0.25, 'parsnip': 0.8, 'peach': 0.65,
        'pear': 0.65, 'pepper': 0.45, 'peppercorn': 0.3, 'pineapple': 0.8, 'plum': 0.7, 'poppy_seed': 0.25,
        'potato': 0.8, 'pumpkin': 0.9, 'purslane': 0.4, 'radicchio': 0.5, 'radish': 0.35, 'rosemary': 0.3,
        'saffron': 0.2, 'sage': 0.3, 'sesame_seed': 0.25, 'sorrel': 0.35, 'spinach': 0.45, 'squash': 0.8,
        'tarragon': 0.25, 'tat_soi': 0.35, 'thyme': 0.3, 'tomato': 0.95, 'turmeric': 0.3, 'turnip': 0.75,
        'vanilla': 0.2, 'watercress': 0.35, 'watermelon': 0.95, 'zucchini': 0.65
    }

    general_type = variety_to_general.get(fruit_type, fruit_type).lower()
    if general_type in weight_factors:
        weight_g = volume_cm3 * weight_factors[general_type]
        return weight_g
    else:
        print(f"No weight data available for {fruit_type}.")
        return None


# Function to estimate nutritional values based on weight
def estimate_nutrition(weight_g, fruit_type):
    general_type = variety_to_general.get(fruit_type, fruit_type).lower()
    # Nutritional values per 100 grams (example values)
    nutrition_data = {
        "anise": {"calories": 337, "protein": 18, "fat": 16, "sugar": 0.0, "cholesterol": 0, "sodium": 16, "calcium": 646, "iron": 37, "potassium": 1441},
        "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "sugar": 10.0, "cholesterol": 0, "sodium": 1, "calcium": 6, "iron": 0.1, "potassium": 107},
        "artichoke": {"calories": 53, "protein": 2.9, "fat": 0.3, "sugar": 1.0, "cholesterol": 0, "sodium": 60, "calcium": 21, "iron": 0.6, "potassium": 286},
        "arugula": {"calories": 25, "protein": 2.6, "fat": 0.7, "sugar": 2.1, "cholesterol": 0, "sodium": 27, "calcium": 160, "iron": 1.5, "potassium": 369},
        "avocado": {"calories": 160, "protein": 2.0, "fat": 15, "sugar": 0.7, "cholesterol": 0, "sodium": 7, "calcium": 12, "iron": 0.6, "potassium": 485},
        "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "sugar": 12, "cholesterol": 0, "sodium": 1, "calcium": 5, "iron": 0.3, "potassium": 358},
        "basil": {"calories": 23, "protein": 3.2, "fat": 0.6, "sugar": 0.3, "cholesterol": 0, "sodium": 4, "calcium": 177, "iron": 3.2, "potassium": 295},
        "bay_leaf": {"calories": 313, "protein": 7.6, "fat": 8.4, "sugar": 0.0, "cholesterol": 0, "sodium": 23, "calcium": 834, "iron": 43, "potassium": 529},
        "beet": {"calories": 44, "protein": 1.7, "fat": 0.2, "sugar": 8, "cholesterol": 0, "sodium": 77, "calcium": 16, "iron": 0.8, "potassium": 305},
        "bok_choy": {"calories": 12, "protein": 1.6, "fat": 0.2, "sugar": 0.8, "cholesterol": 0, "sodium": 34, "calcium": 93, "iron": 1, "potassium": 371},
        "broccoli": {"calories": 35, "protein": 2.4, "fat": 0.4, "sugar": 1.4, "cholesterol": 0, "sodium": 41, "calcium": 40, "iron": 0.7, "potassium": 293},
        "brussels_sprout": {"calories": 36, "protein": 2.6, "fat": 0.5, "sugar": 1.7, "cholesterol": 0, "sodium": 21, "calcium": 36, "iron": 1.2, "potassium": 317},
        "cabbage": {"calories": 23, "protein": 1.3, "fat": 0.1, "sugar": 2.8, "cholesterol": 0, "sodium": 8, "calcium": 48, "iron": 0.2, "potassium": 196},
        "caraway": {"calories": 333, "protein": 19.8, "fat": 14.6, "sugar": 0.6, "cholesterol": 0, "sodium": 0, "calcium": 0, "iron": 0, "potassium": 0},
        "cardamom": {"calories": 311, "protein": 11, "fat": 6.7, "sugar": 0.0, "cholesterol": 0, "sodium": 18, "calcium": 383, "iron": 14, "potassium": 1119},
        "carrot": {"calories": 35, "protein": 0.8, "fat": 0.2, "sugar": 3.5, "cholesterol": 0, "sodium": 58, "calcium": 30, "iron": 0.3, "potassium": 235},
        "cauliflower": {"calories": 23, "protein": 1.8, "fat": 0.5, "sugar": 2.1, "cholesterol": 0, "sodium": 15, "calcium": 16, "iron": 0.3, "potassium": 142},
        "celery": {"calories": 16, "protein": 0.7, "fat": 0.2, "sugar": 1.3, "cholesterol": 0, "sodium": 80, "calcium": 40, "iron": 0.2, "potassium": 260},
        "chard": {"calories": 20, "protein": 1.9, "fat": 0.1, "sugar": 1.1, "cholesterol": 0, "sodium": 179, "calcium": 58, "iron": 2.3, "potassium": 549},
        "chervil": {"calories": 237, "protein": 23, "fat": 3.9, "sugar": 0.0, "cholesterol": 0, "sodium": 83, "calcium": 1346, "iron": 32, "potassium": 4740},
        "chicory": {"calories": 23, "protein": 1.7, "fat": 0.3, "sugar": 0.7, "cholesterol": 0, "sodium": 45, "calcium": 100, "iron": 0.9, "potassium": 420},
        "chive": {"calories": 30, "protein": 3.3, "fat": 0.7, "sugar": 0.6, "cholesterol": 0, "sodium": 0, "calcium": 0, "iron": 0, "potassium": 0},
        "cilantro": {"calories": 23, "protein": 2.1, "fat": 0.5, "sugar": 0.9, "cholesterol": 0, "sodium": 46, "calcium": 67, "iron": 1.8, "potassium": 521},
        "cinnamon": {"calories": 247, "protein": 4, "fat": 1.2, "sugar": 2.2, "cholesterol": 0, "sodium": 10, "calcium": 1002, "iron": 8.3, "potassium": 431},
        "clove": {"calories": 274, "protein": 6.0, "fat": 13.0, "sugar": 0.0, "cholesterol": 0, "sodium": 0, "calcium": 0, "iron": 0, "potassium": 0},
        "collard_greens": {"calories": 33, "protein": 2.7, "fat": 0.7, "sugar": 0.4, "cholesterol": 0, "sodium": 15, "calcium": 141, "iron": 1.1, "potassium": 117},
        "coriander": {"calories": 279, "protein": 22, "fat": 4.8, "sugar": 7.3, "cholesterol": 0, "sodium": 211, "calcium": 1246, "iron": 42, "potassium": 4466},
        "cress": {"calories": 32, "protein": 2.6, "fat": 0.7, "sugar": 0.3, "cholesterol": 0, "sodium": 0, "calcium": 0, "iron": 0, "potassium": 0},
        "cucumber": {"calories": 15, "protein": 0.7, "fat": 0.1, "sugar": 1.7, "cholesterol": 0, "sodium": 2, "calcium": 16, "iron": 0.3, "potassium": 147},
        "cumin": {"calories": 375, "protein": 18, "fat": 22, "sugar": 2.3, "cholesterol": 0, "sodium": 168, "calcium": 931, "iron": 66, "potassium": 1788},
        "dandelion_greens": {"calories": 45, "protein": 2.7, "fat": 0.7, "sugar": 0.7, "cholesterol": 0, "sodium": 76, "calcium": 187, "iron": 3.1, "potassium": 397},
        "dill": {"calories": 43, "protein": 3.5, "fat": 1.1, "sugar": 0.6, "cholesterol": 0, "sodium": 61, "calcium": 208, "iron": 6.6, "potassium": 738},
        "eggplant": {"calories": 35, "protein": 0.8, "fat": 0.2, "sugar": 3.2, "cholesterol": 0, "sodium": 1, "calcium": 6, "iron": 0.3, "potassium": 123},
        "endive": {"calories": 17, "protein": 0.9, "fat": 0.1, "sugar": 0.3, "cholesterol": 0, "sodium": 2, "calcium": 19, "iron": 0.2, "potassium": 211},
        "escarole": {"calories": 19, "protein": 1.2, "fat": 0.2, "sugar": 0.2, "cholesterol": 0, "sodium": 19, "calcium": 46, "iron": 0.7, "potassium": 245},
        "fennel": {"calories": 31, "protein": 1.2, "fat": 0.2, "sugar": 3.9, "cholesterol": 0, "sodium": 52, "calcium": 49, "iron": 0.7, "potassium": 414},
        "fennel_seed": {"calories": 345, "protein": 16, "fat": 15, "sugar": 0.0, "cholesterol": 0, "sodium": 88, "calcium": 1196, "iron": 19, "potassium": 1694},
        "fenugreek": {"calories": 323, "protein": 23, "fat": 6.4, "sugar": 0.0, "cholesterol": 0, "sodium": 67, "calcium": 176, "iron": 34, "potassium": 770},
        "garlic": {"calories": 149, "protein": 6.4, "fat": 0.5, "sugar": 1.0, "cholesterol": 0, "sodium": 17, "calcium": 181, "iron": 1.7, "potassium": 401},
        "ginger": {"calories": 80, "protein": 1.8, "fat": 0.8, "sugar": 1.7, "cholesterol": 0, "sodium": 13, "calcium": 16, "iron": 0.6, "potassium": 415},
        "grapefruit": {"calories": 42, "protein": 0.8, "fat": 0.1, "sugar": 6.9, "cholesterol": 0, "sodium": 0, "calcium": 22, "iron": 0.1, "potassium": 135},
        "kale": {"calories": 28, "protein": 1.9, "fat": 0.4, "sugar": 1.3, "cholesterol": 0, "sodium": 23, "calcium": 72, "iron": 0.9, "potassium": 228},
        "kiwi": {"calories": 61, "protein": 1.1, "fat": 0.5, "sugar": 9, "cholesterol": 0, "sodium": 3, "calcium": 34, "iron": 0.3, "potassium": 312},
        "lettuce": {"calories": 17, "protein": 1.2, "fat": 0.3, "sugar": 1.2, "cholesterol": 0, "sodium": 8, "calcium": 33, "iron": 1, "potassium": 247},
        "lime": {"calories": 30, "protein": 0.7, "fat": 0.2, "sugar": 1.7, "cholesterol": 0, "sodium": 2, "calcium": 33, "iron": 0.6, "potassium": 102},
        "mint": {"calories": 394, "protein": 0, "fat": 0.2, "sugar": 63, "cholesterol": 0, "sodium": 63, "calcium": 3, "iron": 0.3, "potassium": 5},
        "mushroom": {"calories": 28, "protein": 2.2, "fat": 0.5, "sugar": 2.3, "cholesterol": 0, "sodium": 2, "calcium": 6, "iron": 1.7, "potassium": 356},
        "mustard_greens": {"calories": 27, "protein": 2.9, "fat": 0.4, "sugar": 1.3, "cholesterol": 0, "sodium": 20, "calcium": 115, "iron": 1.6, "potassium": 384},
        "okra": {"calories": 22, "protein": 1.9, "fat": 0.2, "sugar": 2.4, "cholesterol": 0, "sodium": 6, "calcium": 77, "iron": 0.3, "potassium": 135},
        "olive": {"calories": 115, "protein": 0.8, "fat": 11, "sugar": 0.0, "cholesterol": 0, "sodium": 735, "calcium": 88, "iron": 3.3, "potassium": 8},
        "onion": {"calories": 44, "protein": 1.4, "fat": 0.2, "sugar": 4.7, "cholesterol": 0, "sodium": 3, "calcium": 22, "iron": 0.2, "potassium": 166},
        "oregano": {"calories": 265, "protein": 9.0, "fat": 4.3, "sugar": 4.1, "cholesterol": 0, "sodium": 25, "calcium": 1597, "iron": 37, "potassium": 1260},
        "parsley": {"calories": 36, "protein": 3.0, "fat": 0.8, "sugar": 0.9, "cholesterol": 0, "sodium": 56, "calcium": 138, "iron": 6.2, "potassium": 554},
        "parsnip": {"calories": 71, "protein": 1.3, "fat": 0.3, "sugar": 4.8, "cholesterol": 0, "sodium": 10, "calcium": 37, "iron": 0.6, "potassium": 367},
        "pea": {"calories": 84, "protein": 5.4, "fat": 0.2, "sugar": 5.9, "cholesterol": 0, "sodium": 3, "calcium": 27, "iron": 1.5, "potassium": 271},
        "peach": {"calories": 39, "protein": 0.9, "fat": 0.3, "sugar": 8.4, "cholesterol": 0, "sodium": 0, "calcium": 6, "iron": 0.3, "potassium": 190},
        "peppermint": {"calories": 70, "protein": 3.8, "fat": 0.9, "sugar": 0.5, "cholesterol": 0, "sodium": 31, "calcium": 243, "iron": 5.1, "potassium": 569},
        "pepper, bell": {"calories": 251, "protein": 10, "fat": 3.3, "sugar": 0.6, "cholesterol": 0, "sodium": 20, "calcium": 443, "iron": 9.7, "potassium": 1329},
        "pepper, chili": {"calories": 43.7, "protein": 1.9, "fat": 0.4, "sugar": 5.1, "cholesterol": 0, "sodium": 0, "calcium": 0, "iron": 0, "potassium": 0},
        "purslane": {"calories": 18, "protein": 1.5, "fat": 0.2, "sugar": 0.0, "cholesterol": 0, "sodium": 44, "calcium": 78, "iron": 0.8, "potassium": 488},
        "radish": {"calories": 16, "protein": 0.7, "fat": 0.1, "sugar": 1.9, "cholesterol": 0, "sodium": 39, "calcium": 25, "iron": 0.3, "potassium": 233},
        "rosemary": {"calories": 331, "protein": 4.9, "fat": 15, "sugar": 0.0, "cholesterol": 0, "sodium": 50, "calcium": 1280, "iron": 29, "potassium": 955},
        "sage": {"calories": 315, "protein": 11, "fat": 13, "sugar": 1.7, "cholesterol": 0, "sodium": 11, "calcium": 1652, "iron": 28, "potassium": 1070},
        "savoy_cabbage": {"calories": 24, "protein": 1.8, "fat": 0.1, "sugar": 1.5, "cholesterol": 0, "sodium": 24, "calcium": 30, "iron": 0.4, "potassium": 184},
        "seaweed": {"calories": 35, "protein": 5.8, "fat": 0.3, "sugar": 0.5, "cholesterol": 0, "sodium": 48, "calcium": 70, "iron": 1.8, "potassium": 356},
        "spinach": {"calories": 23, "protein": 3, "fat": 0.3, "sugar": 0.4, "cholesterol": 0, "sodium": 70, "calcium": 136, "iron": 3.6, "potassium": 466},
        "squash": {"calories": 23, "protein": 1, "fat": 0.4, "sugar": 2.5, "cholesterol": 0, "sodium": 1, "calcium": 22, "iron": 0.4, "potassium": 177},
        "sweet_potato": {"calories": 90, "protein": 2, "fat": 0.2, "sugar": 6.5, "cholesterol": 0, "sodium": 36, "calcium": 38, "iron": 0.7, "potassium": 475},
        "thyme": {"calories": 101, "protein": 5.6, "fat": 1.7, "sugar": 0.0, "cholesterol": 0, "sodium": 9, "calcium": 405, "iron": 17, "potassium": 609},
        "tomato": {"calories": 18, "protein": 0.9, "fat": 0.2, "sugar": 2.6, "cholesterol": 0, "sodium": 5, "calcium": 10, "iron": 0.3, "potassium": 237},
        "turnip": {"calories": 22, "protein": 0.7, "fat": 0.1, "sugar": 3, "cholesterol": 0, "sodium": 16, "calcium": 33, "iron": 0.2, "potassium": 177},
        "watercress": {"calories": 11, "protein": 2.3, "fat": 0.1, "sugar": 0.2, "cholesterol": 0, "sodium": 41, "calcium": 120, "iron": 0.2, "potassium": 330},
        "yam": {"calories": 116, "protein": 1.5, "fat": 0.1, "sugar": 0.5, "cholesterol": 0, "sodium": 8, "calcium": 14, "iron": 0.5, "potassium": 670},
        "zucchini": {"calories": 15, "protein": 1.1, "fat": 0.4, "sugar": 1.7, "cholesterol": 0, "sodium": 3, "calcium": 18, "iron": 0.4, "potassium": 264},
    }

    if general_type in nutrition_data:
        calories = (weight_g / 100) * nutrition_data[general_type]['calories']
        protein = (weight_g / 100) * nutrition_data[general_type]['protein']
        fat = (weight_g / 100) * nutrition_data[general_type]['fat']
        sugar = (weight_g / 100) * nutrition_data[general_type]['sugar']
        cholesterol = (weight_g / 100) * nutrition_data[general_type]['cholesterol']
        sodium = (weight_g / 100) * nutrition_data[general_type]['sodium']
        calcium = (weight_g / 100) * nutrition_data[general_type]['calcium']
        iron = (weight_g / 100) * nutrition_data[general_type]['iron']
        potassium = (weight_g / 100) * nutrition_data[general_type]['potassium']
        return calories, protein, fat, sugar, cholesterol, sodium, calcium, iron, potassium
    else:
        print(f"No nutritional data available for {fruit_type}.")
        return None


# Main function integrating recognition and volume estimation
def get_fruit_nutrition_details(image_path, reference_diameter_mm=24.26):
    # Check if image path exists
    if not os.path.exists(image_path):
        return {'error': f"Image file '{image_path}' does not exist."}

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f"Failed to load image '{image_path}'. Please check the file path and integrity."}

    label, confidence = recognize_food(img)
    general_label = variety_to_general.get(label, label).lower()
    result = {
        'name': general_label,
        'confidence': "{:.2f}".format(confidence) + ' g'
    }

    if confidence > 0.5:
        # Resize and pad image
        resized_img, scale = resize_and_pad(img)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(contour)

        # Reference object measurement
        reference_length_pixels = 50 * scale  # Adjust for resized scale
        pixel_to_mm_ratio = reference_diameter_mm / reference_length_pixels

        # Estimate volume
        volume_mm3 = estimate_volume_based_on_shape(label, area_pixels, pixel_to_mm_ratio)
        volume_cm3 = volume_mm3 / 1000

        # Estimate weight
        weight_g = estimate_weight(volume_cm3, label)
        if weight_g:
            result['estimated_weight'] = "{:.2f}".format(weight_g) + ' g'
            # Estimate nutritional values per 100 g
            nutrition = estimate_nutrition(weight_g, label)
            if nutrition:
                calories, protein, fat, sugar, cholesterol, sodium, calcium, iron, potassium = nutrition
                nutrition_data = {
                    'calories': "{:.2f}".format(calories) + ' kcal',
                    'protein': "{:.2f}".format(protein) + ' g',
                    'fat': "{:.2f}".format(fat) + ' g',
                    'sugar': "{:.2f}".format(sugar) + ' g',
                    'cholesterol': "{:.2f}".format(cholesterol) + ' mg',
                    'sodium': "{:.2f}".format(sodium) + ' mg',
                    'calcium': "{:.2f}".format(calcium) + ' mg',
                    'iron': "{:.2f}".format(iron) + ' mg',
                    'potassium': "{:.2f}".format(potassium) + ' mg'
                }
                result['nutrition_data'] = nutrition_data
        return result
    else:
        result['message'] = "Food recognition confidence too low for volume estimation."
        return result


# if __name__ == "__main__":
#     get_fruit_nutrition_details('orange.jpg')  # Replace with your image path
