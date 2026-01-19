import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR   = os.path.join(BASE_DIR, "embeddings")
TEST_DIR = os.path.join(BASE_DIR, "data_all", "test_inputs")
OUT_DIR  = os.path.join(BASE_DIR, "outputs_extreme")

os.makedirs(OUT_DIR, exist_ok=True)

EMB_NPY = os.path.join(DB_DIR, "embeddings.npy")
LBL_NPY = os.path.join(DB_DIR, "labels.npy")

# ======================================================
# RECOGNITION PARAMETERS 
# ======================================================
SIM_THRESHOLD  = 0.20
WEAK_THRESHOLD = 0.10
HYSTERESIS_LOW = 0.38

QUALITY_BONUS_MAX = 0.18
TEMPORAL_ALPHA    = 0.7

# ======================================================
# DETECTION PARAMETERS 
# ======================================================
SCALE_FACTORS = [0.7, 0.9, 1.0, 1.15, 1.3]
GAMMA_VALUES  = [0.6, 0.8, 1.0, 1.2, 1.5]

IOU_THRESHOLD = 0.55
MIN_FACE_SIZE = 35

# ======================================================
# LOAD DATABASE
# ======================================================
print("[INFO] Loading face database...")
db_embeddings = np.load(EMB_NPY)
db_labels = np.load(LBL_NPY)

db_embeddings = db_embeddings / np.linalg.norm(
    db_embeddings, axis=1, keepdims=True
)

print(f"[INFO] Loaded {len(db_labels)} identities")

# ======================================================
# INIT SCRFD + ARCFACE (BUFFALO_L)
# ======================================================
print("[INFO] Initializing SCRFD + ArcFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(1024, 1024))

# ======================================================
# IMAGE ENHANCEMENT
# ======================================================
def gamma_correct(img, gamma):
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# ======================================================
# IOU
# ======================================================
def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

# ======================================================
# SOFT-NMS 
# ======================================================
def soft_nms_faces(faces, iou_thresh=0.55, score_decay=0.7):
    if not faces:
        return []

    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
    kept = []

    while faces:
        best = faces.pop(0)
        kept.append(best)

        survivors = []
        for f in faces:
            overlap = iou(best.bbox, f.bbox)
            if overlap > iou_thresh:
                f.det_score *= score_decay
            if f.det_score > 0.15:
                survivors.append(f)

        faces = sorted(survivors, key=lambda f: f.det_score, reverse=True)

    return kept

# ======================================================
# TEMPORAL MEMORY
# ======================================================
identity_memory = defaultdict(lambda: deque(maxlen=10))

# ======================================================
# PROCESS IMAGES
# ======================================================
for img_name in sorted(os.listdir(TEST_DIR)):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    print(f"\n[PROCESSING] {img_name}")
    img_path = os.path.join(TEST_DIR, img_name)

    base_img = cv2.imread(img_path)
    if base_img is None:
        continue

    candidates = []

    # ===============================
    # MULTI-SCALE + MULTI-GAMMA
    # ===============================
    for scale in SCALE_FACTORS:
        h, w = base_img.shape[:2]
        resized = cv2.resize(base_img, (int(w*scale), int(h*scale)))

        for gamma in GAMMA_VALUES:
            img = gamma_correct(resized, gamma)
            img = apply_clahe(img)
            img = denoise(img)

            faces = app.get(img)

            for f in faces:
                f.bbox /= scale
                candidates.append(f)

    # ===============================
    # RESCUE PASS 
    # ===============================
    if len(candidates) < 5:
        rescue = cv2.resize(base_img, None, fx=1.5, fy=1.5)
        rescue_faces = app.get(rescue)

        for f in rescue_faces:
            f.bbox /= 1.5
            f.det_score *= 0.85
            candidates.append(f)

    # ===============================
    # SOFT-NMS
    # ===============================
    candidates = soft_nms_faces(candidates, IOU_THRESHOLD)
    print(f"[DETECTED] {len(candidates)} faces")

    # ===============================
    # RECOGNITION
    # ===============================
    for face in candidates:
        x1, y1, x2, y2 = face.bbox.astype(int)

        w = x2 - x1
        h = y2 - y1

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            if face.det_score < 0.45:
                continue

        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        sims = cosine_similarity(
            emb.reshape(1, -1), db_embeddings
        )[0]

        idx = np.argmax(sims)
        score = sims[idx]
        label = db_labels[idx]

        # Temporal smoothing
        memory = identity_memory[label]
        memory.append(score)
        smoothed = TEMPORAL_ALPHA * np.mean(memory) + (1-TEMPORAL_ALPHA) * score

        if smoothed >= SIM_THRESHOLD:
            color = (0,255,0)
            text = f"{label} ({smoothed:.3f})"
        elif score >= WEAK_THRESHOLD:
            color = (0,165,255)
            text = f"{label}? ({score:.3f})"
        else:
            color = (0,0,255)
            text = "UNKNOWN"

        cv2.rectangle(base_img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            base_img, text, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, base_img)
    print(f"SAVED → {out_path}")

print("\n✅ EXTREME DETECTION + RECOGNITION COMPLETED")
