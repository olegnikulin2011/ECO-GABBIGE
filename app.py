# Сервер ECObage.
#
# Как распознаём объект:
# 1. YOLO смотрит на фото и говорит, что на нём вообще есть (bottle, apple, banana...).
# 2. Если YOLO уверена (confidence >= 0.5), по табличке YOLO_DIRECT_MAP определяем
#    категорию сортировки. Для пары объектов с неоднозначным материалом (bottle, cup)
#    дополнительно спрашиваем нашу модель, обученную на датасете мусора, чтобы выбрать
#    конкретный материал - но сам факт "что это за объект" всегда решает YOLO.
# 3. Если YOLO не нашла ничего подходящего (мусора на фото часто нет в 80 классах COCO -
#    там нет ни банки, ни картона, ни пакета) - подключаем как запасной вариант нашу
#    модель, обученную на датасете мусора, и берём её лучший класс.
# 4. Если и она не уверена - честно говорим, что не смогли распознать объект,
#    вместо того чтобы гадать.

import json

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image, ImageOps
from ultralytics import YOLO

app = Flask(__name__, static_folder=".", static_url_path="")

YOLO_CONFIDENCE = 0.35
DATASET_CONFIDENCE = 0.35

# YOLO-класс (из набора COCO) -> наш класс отходов. Дальше class_name сопоставляется
# с категорией сортировки уже на фронтенде (см. CLASS_MAP в app.js).
YOLO_DIRECT_MAP = {
    "apple": "Apple",
    "banana": "Ordure_menagere",
    "orange": "Ordure_menagere",
    "broccoli": "Ordure_menagere",
    "carrot": "Ordure_menagere",
    "sandwich": "Ordure_menagere",
    "hot dog": "Ordure_menagere",
    "pizza": "Ordure_menagere",
    "donut": "Ordure_menagere",
    "cake": "Ordure_menagere",
    "wine glass": "Verre",
    "vase": "Verre",
    "book": "Papier_Carton",
}

# у этих YOLO-классов материал неоднозначный, поэтому его уточняет наша модель,
# обученная на датасете мусора - выбираем более вероятный из вариантов
YOLO_NEEDS_MATERIAL_CHECK = {
    "bottle": ["Bouteille_plastique", "Verre"],
    "cup": ["Bouteille_plastique", "Papier_Carton", "Emballage_metallique"],
}

# YOLO иногда вообще не видит фрукт/еду на фото (например, надкушенное яблоко без
# кожуры плохо похоже на "apple"). Второй, независимый источник - обычный ImageNet:
# среди его 1000 классов много конкретных фруктов, овощей и грибов, и если хотя бы
# один из них попадает в топ-5 - считаем, что на фото органика.
FOOD_IMAGENET_CLASSES = {
    "granny smith", "banana", "orange", "lemon", "strawberry", "pineapple",
    "fig", "pomegranate", "artichoke", "cucumber", "zucchini", "bell pepper",
    "corn", "cauliflower", "broccoli", "head cabbage", "spaghetti squash",
    "acorn squash", "butternut squash", "custard apple", "jackfruit",
    "bolete", "agaric",
}
FOOD_FLOOR = 0.2

# --- модели ---

yolo_model = YOLO("model/yolov8n.pt")

with open("model/classes.json", encoding="utf-8") as f:
    dataset_classes = json.load(f)

dataset_backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
dataset_backbone.eval()

dataset_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, len(dataset_classes)))
dataset_head.load_state_dict(torch.load("model/head.pt", map_location="cpu"))
dataset_head.eval()

imagenet_classes = MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]

dataset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- детекция объекта через YOLO ---
# Смотрим не только на самый уверенный бокс, а на все найденные объекты и берём
# среди них лучший, который вообще есть в наших табличках. Так фото вроде надкусанного
# яблока не теряется из-за того, что YOLO уверена в "cake" сильнее, чем в "apple".
def detect_object(image):
    results = yolo_model.predict(image, verbose=False, conf=0.1)[0]

    best_label, best_conf = None, 0.0
    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if label not in YOLO_DIRECT_MAP and label not in YOLO_NEEDS_MATERIAL_CHECK:
            continue
        if confidence > best_conf:
            best_label, best_conf = label, confidence

    return best_label, best_conf


# --- запасной способ узнать еду: общий ImageNet (не наш датасет) ---
def detect_food(image):
    x = dataset_transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(dataset_backbone(x), dim=1)[0]

    top5 = torch.topk(probs, 5)
    for prob, idx in zip(top5.values, top5.indices):
        name = imagenet_classes[idx].lower()
        if name in FOOD_IMAGENET_CLASSES and prob >= FOOD_FLOOR:
            return name, float(prob)

    return None, 0.0


# --- наша модель, обученная на датасете мусора (архив archive.zip) ---
def dataset_probs(image):
    x = dataset_transform(image).unsqueeze(0)
    with torch.no_grad():
        features = dataset_backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        probs = torch.softmax(dataset_head(features), dim=1)[0]
    return {name: float(probs[i]) for i, name in enumerate(dataset_classes)}


# выбирает более вероятный материал из нескольких вариантов (для bottle/cup)
def pick_material(image, candidates):
    probs = dataset_probs(image)
    return max(candidates, key=lambda name: probs[name])


# запасной вариант, когда YOLO не нашла ничего подходящего: берём лучший класс датасетной модели
def classify_with_dataset(image):
    probs = dataset_probs(image)
    best_name = max(probs, key=probs.get)
    return best_name, probs[best_name]


# --- объект -> класс для сортировки ---
def classify(image):
    label, confidence = detect_object(image)

    if label is not None and confidence >= YOLO_CONFIDENCE:
        if label in YOLO_DIRECT_MAP:
            return YOLO_DIRECT_MAP[label], confidence

        if label in YOLO_NEEDS_MATERIAL_CHECK:
            material = pick_material(image, YOLO_NEEDS_MATERIAL_CHECK[label])
            return material, confidence

        # YOLO уверенно нашла объект, но это не то, что мы сортируем (человек, машина и т.п.)
        return "Unknown", 0.0

    # YOLO не нашла еду/фрукт - на всякий случай спрашиваем ещё общий ImageNet
    # (иногда он видит фрукт там, где YOLO уверена в чём-то постороннем)
    food_name, food_conf = detect_food(image)
    if food_name is not None:
        return ("Apple" if food_name == "granny smith" else "Ordure_menagere"), food_conf

    # и только после этого - запасной вариант через нашу модель, обученную на датасете мусора
    name, probability = classify_with_dataset(image)
    if probability >= DATASET_CONFIDENCE:
        return name, probability

    return "Unknown", 0.0


# --- API ---
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    image_file = request.files["image"]
    image = Image.open(image_file.stream)
    image = ImageOps.exif_transpose(image).convert("RGB")  # телефоны пишут поворот в EXIF, Pillow сам не поворачивает

    class_name, probability = classify(image)
    return jsonify({"class_name": class_name, "probability": probability})


if __name__ == "__main__":
    app.run(port=8765, debug=True)
