import os
import io
import cv2
import json
import base64
import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from flask import Flask, request, jsonify
from keras_facenet import FaceNet
import joblib
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
USE_CNN = True
DETECTOR = "mtcnn"
TARGET_FACE_SIZE = (160, 160)
DISPLAY_FRAME_SIZE = (320, 240)
DEFAULT_THRESHOLD_FALLBACK = 0.60
SPOOF_MODE = "torch_yolo"
YOLO_IMG_SIZE = 640
YOLO_MIN_CONF = 0.10
SPOOF_THRESHOLD = 0.10

BASE_DIR = "/app"
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(EMB_DIR, "face_classifier.pkl")
ENCODER_PATH    = os.path.join(EMB_DIR, "label_encoder.pkl")
META_PATH       = os.path.join(EMB_DIR, "face_svm_meta.json")
CNN_MODEL_PATH  = os.path.join(EMB_DIR, "face_cnn_best.keras")
YOLO_SPOOF_PATH = os.path.join(EMB_DIR, "best10-m-ObjDet-robo.pt")

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
log_path = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# GOOGLE DRIVE AUTO-SYNC (Service Account)
# ------------------------------------------------------------------------------
def sync_from_drive():
    """Download models from Google Drive if missing or forced."""
    import gdown, tempfile

    logging.info("🔄 Checking Google Drive for model updates...")

    ASSET_IDS = {
        "face_classifier.pkl": os.getenv("GDRIVE_CLASSIFIER_ID", ""),
        "label_encoder.pkl":   os.getenv("GDRIVE_ENCODER_ID",  ""),
        "face_svm_meta.json":  os.getenv("GDRIVE_META_ID",     ""),
        "face_cnn_best.keras": os.getenv("GDRIVE_CNN_ID",      ""),
        "best10-m-ObjDet-robo.pt": os.getenv("GDRIVE_YOLO_ID", "")
    }

    force_refresh = os.getenv("FORCE_REFRESH_MODELS", "0") == "1"
    svc_json = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", None)

    if not any(ASSET_IDS.values()):
        logging.warning("No Google Drive file IDs set. Skipping sync.")
        return

    creds_file = None
    if svc_json:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(svc_json.encode())
            creds_file = f.name

    for fname, fid in ASSET_IDS.items():
        if not fid:
            continue
        local_path = os.path.join(EMB_DIR, fname)
        if os.path.exists(local_path) and not force_refresh:
            logging.info(f"[Drive] {fname} up-to-date.")
            continue
        try:
            url = f"https://drive.google.com/uc?id={fid}"
            logging.info(f"[Drive] Updating {fname} ...")
            gdown.download(url, local_path, quiet=False)
            logging.info(f"[Drive] {fname} refreshed.")
        except Exception as e:
            logging.warning(f"[Drive] Failed to fetch {fname}: {e}")

    if creds_file and os.path.exists(creds_file):
        os.remove(creds_file)

# Run sync before loading models
sync_from_drive()

# ------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------
embedder = FaceNet()
classifier = joblib.load(CLASSIFIER_PATH)

if os.path.exists(META_PATH):
    meta = json.load(open(META_PATH, "r"))
    CLASSES = meta.get("classes", [])
    META_THRESHOLD = float(meta.get("threshold", DEFAULT_THRESHOLD_FALLBACK))
    logging.info(f"Loaded meta: {len(CLASSES)} classes, threshold={META_THRESHOLD:.2f}")
else:
    CLASSES, META_THRESHOLD = [], DEFAULT_THRESHOLD_FALLBACK
    logging.warning("Meta file not found; using default threshold.")

label_encoder = None
if os.path.exists(ENCODER_PATH):
    label_encoder = joblib.load(ENCODER_PATH)
    if not CLASSES and hasattr(label_encoder, "classes_"):
        CLASSES = list(label_encoder.classes_)
        logging.info(f"Classes from label_encoder: {len(CLASSES)}")
else:
    logging.warning("Label encoder not found; proceeding anyway.")

cnn_model = None
if USE_CNN:
    try:
        cnn_model = load_model(CNN_MODEL_PATH)
        logging.info("CNN model loaded.")
    except Exception as e:
        logging.warning(f"Could not load CNN model: {e}. Using SVM only.")
        USE_CNN = False

if DETECTOR.lower() == "haar":
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    try:
        from mtcnn import MTCNN
        mtcnn = MTCNN()
        logging.info("Using MTCNN for face detection.")
    except Exception as e:
        logging.warning(f"MTCNN unavailable ({e}); falling back to Haar.")
        DETECTOR = "haar"
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def decode_base64_image(image_data: str) -> Image.Image:
    if 'base64,' in image_data:
        image_data = image_data.split('base64,', 1)[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    return Image.open(image_bytes).convert('RGB')

def preprocess_frame(pil_image: Image.Image) -> np.ndarray:
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, DISPLAY_FRAME_SIZE)
    return cv2.flip(frame, 1)

def detect_and_crop_face(image_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int,int,int,int]]]:
    if DETECTOR.lower() == "mtcnn":
        dets = mtcnn.detect_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not dets:
            return image_bgr, None
        det = max(dets, key=lambda d: d['box'][2]*d['box'][3])
        x, y, w, h = det['box']
        m = 10
        x = max(0, x-m); y = max(0, y-m)
        w = min(image_bgr.shape[1]-x, w+2*m)
        h = min(image_bgr.shape[0]-y, h+2*m)
        crop = cv2.resize(image_bgr[y:y+h, x:x+w], TARGET_FACE_SIZE)
        return crop, (int(x), int(y), int(w), int(h))
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(30,30))
        if len(faces)==0:
            return image_bgr, None
        (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
        crop = cv2.resize(image_bgr[y:y+h, x:x+w], TARGET_FACE_SIZE)
        return crop, (int(x), int(y), int(w), int(h))

def prewhiten(x: np.ndarray) -> np.ndarray:
    m, s = x.mean(), x.std()
    s = max(s, 1.0 / np.sqrt(x.size))
    return (x - m) / s

def face_to_embedding(cropped_face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB)
    emb = embedder.embeddings([face_rgb])[0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)

def idx_to_label(idx: int) -> str:
    if 0 <= idx < len(CLASSES):
        return CLASSES[idx]
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([idx])[0])
        except Exception:
            pass
    return str(idx)

def predict_svm_label_conf(embedding: np.ndarray, threshold: float) -> Tuple[str,float]:
    sample = embedding.reshape(1, -1)
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(sample)[0]
        conf = float(proba.max())
        pred_idx = int(proba.argmax())
    else:
        pred_idx = int(classifier.predict(sample)[0])
        conf = 0.0
    label = idx_to_label(pred_idx)
    if conf < threshold:
        label = "Unauthorized"
    return label, conf

def cnn_predict(cropped_face_bgr: np.ndarray, threshold: float) -> Tuple[str,float]:
    if cnn_model is None:
        return "Unauthorized", 0.0
    face_rgb = cv2.cvtColor(cv2.resize(cropped_face_bgr, (224,224)), cv2.COLOR_BGR2RGB)
    face_norm = face_rgb.astype("float32") / 255.0
    preds = cnn_model.predict(np.expand_dims(face_norm, axis=0), verbose=0)[0]
    conf = float(np.max(preds))
    label = idx_to_label(int(np.argmax(preds)))
    if conf < threshold:
        label = "Unauthorized"
    return label, conf

# ------------------------------------------------------------------------------
# YOLO LIVENESS
# ------------------------------------------------------------------------------
yolo_model = None
if SPOOF_MODE.lower() == "torch_yolo":
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_SPOOF_PATH)
        logging.info("YOLO anti-spoof model loaded.")
    except Exception as e:
        logging.warning(f"YOLO anti-spoof unavailable ({e}); skipping liveness.")
        SPOOF_MODE = "off"

def yolo_liveness_score(image_bgr: np.ndarray) -> float:
    if yolo_model is None:
        return 0.5
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=rgb, imgsz=YOLO_IMG_SIZE, conf=YOLO_MIN_CONF, verbose=False)
    if not results or len(results)==0:
        return 0.5
    names = getattr(yolo_model, "names", {})
    res = results[0]
    confs = res.boxes.conf.cpu().numpy().astype(float) if res.boxes is not None else []
    clsids = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
    spoof_conf = 0.0; face_conf = 0.0
    for c, s in zip(clsids, confs):
        name = names.get(int(c), str(int(c))).lower()
        if name in {"device","photo","screen","fake"}:
            spoof_conf = max(spoof_conf, s)
        if name in {"face","live","real"}:
            face_conf = max(face_conf, s)
    live_score = face_conf * (1.0 - spoof_conf)
    return float(np.clip(live_score, 0.0, 1.0))

# ------------------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/healthz")
def healthz():
    return jsonify({"status":"ok","message":"FaceNet API running"}), 200

@app.route("/classify-face", methods=["POST"])
def classify_face():
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error":"No image provided"}),400

        threshold = float(data.get("threshold", META_THRESHOLD))
        spoof_threshold = float(data.get("spoof_threshold", SPOOF_THRESHOLD))

        pil_image = decode_base64_image(data["image"])
        frame = preprocess_frame(pil_image)
        cropped_face, bbox = detect_and_crop_face(frame)

        # Liveness
        live_score = yolo_liveness_score(frame) if SPOOF_MODE=="torch_yolo" else 0.5
        if live_score < spoof_threshold:
            return jsonify({
                "prediction":{"label":"Unauthorized","confidence":0.0,"reason":"spoof_detected"},
                "liveness":{"score":round(live_score,3),"threshold":spoof_threshold}
            }),200

        embedding = face_to_embedding(cropped_face)
        svm_label, svm_conf = predict_svm_label_conf(embedding, threshold)

        if USE_CNN:
            cnn_label, cnn_conf = cnn_predict(cropped_face, threshold)
            if svm_conf >= cnn_conf:
                final_label, final_conf, model_used = svm_label, svm_conf, "facenet_svm"
            else:
                final_label, final_conf, model_used = cnn_label, cnn_conf, "cnn"
        else:
            final_label, final_conf, model_used = svm_label, svm_conf, "facenet_svm"

        return jsonify({
            "prediction":{"label":final_label,"confidence":round(final_conf,4),"model_used":model_used},
            "facenet":{"label":svm_label,"confidence":round(svm_conf,4)},
            "cnn":{"label":cnn_label,"confidence":round(cnn_conf,4)} if USE_CNN else None,
            "liveness":{"score":round(live_score,3),"threshold":spoof_threshold}
        }),200

    except Exception as e:
        logging.exception("Error during classification")
        return jsonify({"error":str(e)}),500

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
