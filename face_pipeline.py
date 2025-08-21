# face_pipeline.py
# RetinaFace (MobileNet-0.25) detector + pluggable embedding head:
# - Default: MobileNetV2 head (PyTorch, 128-D, L2-normalized)
# - Optional: MobileFaceNet (ArcFace) via InsightFace (better identity embeddings)
#
# Works with biubug6/Pytorch_Retinaface and your mobilenet0.25_Final.pth.

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# --- RetinaFace imports (repo root) ---
from data import cfg_mnet
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm

# Try repo's CPU NMS; fallback to torchvision NMS
try:
    from utils.nms.py_cpu_nms import py_cpu_nms as nms_cpu

    _USE_TV_NMS = False
except Exception:
    import torchvision

    _USE_TV_NMS = True

# Optional: InsightFace for MobileFaceNet(+ArcFace)
try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model as if_get_model

    _HAS_INSIGHTFACE = True
except Exception:
    _HAS_INSIGHTFACE = False
    FaceAnalysis = None
    if_get_model = None


# --------------------------
# Utils
# --------------------------
def _check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    print(f"Loaded {len(used_pretrained_keys)} / {len(model_keys)} keys from checkpoint.")


def _remove_prefix(state_dict, prefix):
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(k): v for k, v in state_dict.items()}


def load_retinaface_mnet(weights_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')
    weights_path = str(weights_path)
    assert os.path.isfile(weights_path), f"RetinaFace weights not found: {weights_path}"

    print(f"[Detector] Loading weights: {weights_path}")
    if device.type == "cpu":
        pretrained_dict = torch.load(weights_path, map_location="cpu")
    else:
        pretrained_dict = torch.load(weights_path, map_location=lambda s, l: s.cuda(device.index))

    if "state_dict" in pretrained_dict:
        pretrained_dict = _remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = _remove_prefix(pretrained_dict, "module.")
    _check_keys(net, pretrained_dict)
    net.load_state_dict(pretrained_dict, strict=False)
    net.to(device).eval()
    return net, cfg


def clip_box_to_image(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return np.array([x1, y1, x2, y2], dtype=np.int32)


# --------------------------
# Detector Wrapper
# --------------------------
class RetinaFaceDetector:
    """
    RetinaFace (MobileNet 0.25). Returns list of dicts:
    {"bbox": [x1,y1,x2,y2], "score": float, "landmarks": np.ndarray shape (5,2)}
    """

    def __init__(
            self,
            weights_path: str,
            device: str | torch.device = None,
            conf_thresh: float = 0.6,
            nms_thresh: float = 0.4,
            top_k: int = 1000,
            keep_top_k: int = 500
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.net, self.cfg = load_retinaface_mnet(weights_path, self.device)

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k
        self.keep_top_k = keep_top_k

    @torch.no_grad()
    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "Expected HxWx3 BGR image."
        img = np.float32(img_bgr.copy())
        im_h, im_w, _ = img.shape

        # Preprocess (BGR mean-sub)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)  # CHW
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Forward
        loc, conf, landms = self.net(x)  # [1,N,4], [1,N,2], [1,N,10]

        # Decode boxes
        priors = PriorBox(self.cfg, image_size=(im_h, im_w)).forward().to(self.device)
        boxes = decode(loc.squeeze(0), priors, self.cfg['variance'])
        boxes = boxes * torch.tensor([im_w, im_h, im_w, im_h], device=self.device)

        # Decode landmarks (5 points)
        landms = decode_landm(landms.squeeze(0), priors, self.cfg['variance'])
        landms = landms * torch.tensor([im_w, im_h] * 5, device=self.device)

        scores = conf.squeeze(0)[:, 1]

        # Filter by score
        inds = torch.where(scores > self.conf_thresh)[0]
        boxes, scores, landms = boxes[inds], scores[inds], landms[inds]
        if boxes.numel() == 0:
            return []

        # Sort + top_k
        scores, order = scores.sort(descending=True)
        order = order[:self.top_k]
        boxes, landms = boxes[order], landms[order]

        # Build dets for NMS: [x1,y1,x2,y2,score]
        dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1).detach().cpu().numpy()
        lms = landms.detach().cpu().numpy().reshape(-1, 5, 2)

        # NMS
        if _USE_TV_NMS:
            tb = torch.tensor(dets[:, :4])
            ts = torch.tensor(dets[:, 4])
            keep_idx = torch.ops.torchvision.nms(tb, ts, self.nms_thresh).numpy().tolist()
        else:
            keep_idx = nms_cpu(dets, self.nms_thresh)

        dets = dets[keep_idx][:self.keep_top_k]
        lms = lms[keep_idx][:len(dets)]

        # Build results
        results = []
        for i in range(len(dets)):
            x1, y1, x2, y2, s = dets[i]
            bbox = clip_box_to_image(np.array([x1, y1, x2, y2]), im_w, im_h)
            results.append({
                "bbox": bbox.tolist(),
                "score": float(s),
                "landmarks": lms[i].astype(np.float32)  # (5,2)
            })
        return results


# --------------------------
# 5-point aligner (ArcFace template)
# --------------------------
class FaceAligner:
    """
    Align to 112x112 using ArcFace 5-point template: (L-eye, R-eye, nose, L-mouth, R-mouth)
    """

    def __init__(self, out_size: int = 112):
        self.ref = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        self.out_size = out_size

    def align(self, img_bgr: np.ndarray, landmarks_5x2: np.ndarray) -> np.ndarray:
        src = landmarks_5x2.astype(np.float32)
        dst = self.ref.copy()
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img_bgr, M, (self.out_size, self.out_size), flags=cv2.INTER_LINEAR)
        return aligned


# --------------------------
# Embedding backends
# --------------------------
class MobileNetV2Head(nn.Module):
    """Lightweight generic embedding head (128-D) using ImageNet MobileNetV2 backbone."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class InsightFaceEmbedder:
    """
    Loads either:
      - an InsightFace *pack* (buffalo_l/m/s, buffalo_sc, antelopev2) via FaceAnalysis
      - or a direct model name/path via model_zoo.get_model (e.g., a .onnx file path)

    Expects aligned 112x112 BGR; returns an L2-normalized embedding.
    """
    def __init__(self, device: torch.device, model_id: str = "buffalo_l", input_size: int = 112):
        if not _HAS_INSIGHTFACE:
            raise ImportError("insightface not installed. pip install insightface onnxruntime")
        self.input_size = input_size
        ctx_id = -1 if device.type == "cpu" else 0

        pack_names = {"buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"}
        use_pack = (model_id in pack_names) and (FaceAnalysis is not None)

        if use_pack:
            app = FaceAnalysis(name=model_id)
            app.prepare(ctx_id=ctx_id)
            rec = app.models.get("recognition", None)
            if rec is None:
                raise RuntimeError(f"Recognition model missing in pack '{model_id}'.")
            self.model = rec
        else:
            # treat as model name or local .onnx path
            self.model = if_get_model(model_id)
            if self.model is None:
                raise RuntimeError(
                    f"InsightFace could not load '{model_id}'. "
                    f"Use a pack like 'buffalo_l' or a valid ONNX path."
                )
            self.model.prepare(ctx_id=ctx_id)

    def embed_bgr_112(self, img_bgr_112: np.ndarray) -> np.ndarray:
        # InsightFace recognizers expect RGB 112x112
        img_rgb = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
        m = self.model
        emb = None
        last_err = None

        # Try known API variants across InsightFace versions
        for fn in (
                lambda: m.get(img_rgb),  # some builds
                lambda: m.get(face=img_rgb),  # some builds require kw
                lambda: m.get_feat(img_rgb),  # other builds
                lambda: m.get_feat(face=img_rgb),  # kw form
                lambda: m.compute_embedding(img_rgb),  # rarer
                lambda: m(img_rgb),  # callable model
        ):
            try:
                emb = fn()
                break
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        if emb is None:
            raise RuntimeError(
                f"Could not call recognizer; got {type(m).__name__}. "
                f"Attrs: {[a for a in dir(m) if not a.startswith('_')]} "
                f"Last error: {last_err}"
            )

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(emb) + 1e-9
        return emb / n


class FaceFeatureExtractor:
    """
    Feature extractor with optional InsightFace (ArcFace) backend.
    - backend="insightface": MobileFaceNet/ArcFace via InsightFace (aligned, 112x112)
    - backend="mobilenetv2": default PyTorch MobileNetV2 head (generic, 128-D)
    """

    def __init__(self,
                 device: str | torch.device = None,
                 backend: str = "mobilenetv2",
                 out_dim: int = 128,
                 input_size: int = 112,
                 insightface_model_name: str = "mobilefacenet"):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.backend = backend.lower()
        self.input_size = input_size
        self.aligner = FaceAligner(out_size=input_size)

        if self.backend == "insightface":
            if not _HAS_INSIGHTFACE:
                print("[Feature] insightface not found; falling back to mobilenetv2 head.")
                self.backend = "mobilenetv2"
            else:
                self.iface = InsightFaceEmbedder(self.device,
                                                 model_id=insightface_model_name,
                                                 input_size=input_size)
                self.tf = None  # handled by InsightFace path

        if self.backend == "mobilenetv2":
            self.model = MobileNetV2Head(out_dim=out_dim).to(self.device).eval()
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((input_size, input_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @torch.no_grad()
    def extract_one(self, img_bgr: np.ndarray, bbox: List[int], landmarks_5x2: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns (D,) embedding. If landmarks are available, performs 5-pt alignment to input_size (default 112).
        """
        # Prefer alignment (especially for ArcFace models)
        if landmarks_5x2 is not None:
            face_112 = self.aligner.align(img_bgr, landmarks_5x2)
        else:
            x1, y1, x2, y2 = bbox
            crop = img_bgr[y1:y2, x1:x2, :]
            if crop.size == 0:
                return np.zeros(128, dtype=np.float32)
            face_112 = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        if self.backend == "insightface":
            return self.iface.embed_bgr_112(face_112)

        # mobilenetv2 path
        face_rgb = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
        x = self.tf(face_rgb).unsqueeze(0).to(self.device)
        feat = self.model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return feat


# --------------------------
# Rendering with temporary IDs
# --------------------------
def draw_detections_with_ids(img_bgr: np.ndarray, detections: List[Dict], seed: int | None = None) -> np.ndarray:
    if seed is not None:
        random.seed(seed)

    vis = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = random.randint(1000, 9999)  # temporary
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return vis


# --------------------------
# Pipeline helper (CSV writer)
# --------------------------
def process_image(
        image_path: str,
        detector: RetinaFaceDetector,
        extractor: FaceFeatureExtractor,
        csv_out: str,
        image_out: str | None = None,
) -> List[Dict]:
    """
    1) Detect -> boxes + 5-pt landmarks
    2) For each box, align (if landmarks), extract embedding
    3) Write CSV: x1,y1,x2,y2,feat_0...feat_(D-1)
    4) Optionally save visualization with temp IDs
    """
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Failed to read image: {image_path}"

    detections = detector.detect(img_bgr)

    rows = []
    for det in detections:
        bbox = det["bbox"]
        landmarks = det.get("landmarks", None)
        feat = extractor.extract_one(img_bgr, bbox, landmarks)
        row = {"left_upper_x": bbox[0], "left_upper_y": bbox[1], "right_lower_x": bbox[2], "right_lower_y": bbox[3]}
        row.update({f"feat_{i}": float(feat[i]) for i in range(len(feat))})
        rows.append(row)

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    if image_out is not None:
        vis = draw_detections_with_ids(img_bgr, detections)
        Path(image_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_out, vis)

    return detections


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    """
    Examples:
      # Default backend (mobilenetv2, PyTorch):
      python3 face_pipeline.py \
        --weights ./weights/mobilenet0.25_Final.pth \
        --image ./samples/sample.jpg \
        --csv_out ./outputs/test_features.csv \
        --img_out ./outputs/test_vis.jpg

      # ArcFace via InsightFace pack (recommended):
      python3 face_pipeline.py \
      --weights ./weights/mobilenet0.25_Final.pth \
      --image ./samples/sample.jpg \
      --csv_out ./outputs/test_features.csv \
      --img_out ./outputs/test_vis.jpg \
      --backend insightface \
      --insightface_model buffalo_l
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to mobilenet0.25_Final.pth")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_out", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--img_out", type=str, default=None, help="Optional path to save visualization")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if None)")
    parser.add_argument("--conf", type=float, default=0.6, help="Detector confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="Detector NMS threshold")

    # Embedding backend knobs
    parser.add_argument("--backend", choices=["mobilenetv2", "insightface"], default="mobilenetv2",
                        help="Embedding backend (default mobilenetv2).")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dim for mobilenetv2 backend")
    parser.add_argument("--input_size", type=int, default=112, help="Aligned face size (ArcFace default 112)")
    parser.add_argument("--insightface_model", type=str, default="buffalo_l",
                        help="InsightFace pack name (buffalo_l/m/s, buffalo_sc, antelopev2) or a local .onnx path")

    args = parser.parse_args()

    detector = RetinaFaceDetector(
        weights_path=args.weights,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
    )

    extractor = FaceFeatureExtractor(
        device=args.device,
        backend=args.backend,
        out_dim=args.embed_dim,
        input_size=args.input_size,
        insightface_model_name=args.insightface_model
    )

    dets = process_image(
        image_path=args.image,
        detector=detector,
        extractor=extractor,
        csv_out=args.csv_out,
        image_out=args.img_out
    )
    print(f"Detections: {len(dets)}")
    for d in dets:
        print(d)
