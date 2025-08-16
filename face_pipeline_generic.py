# face_pipeline_v1.py
# Lightweight face detection (RetinaFace-MobileNet0.25) + lightweight feature extractor (MobileNetV2 head)
# Works with biubug6/Pytorch_Retinaface and mobilenet0.25_Final.pth.

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# --- RetinaFace imports (expect this file to live in the repo root) ---
# Repo: https://github.com/biubug6/Pytorch_Retinaface
from data import cfg_mnet
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm

# Try CPU NMS from the repo only; if missing, fallback to default torchvision.ops.nms
try:
    from utils.nms.py_cpu_nms import py_cpu_nms as nms_cpu

    _USE_TV_NMS = False
except Exception:
    import torchvision

    _USE_TV_NMS = True


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
    """
    Build RetinaFace (MobileNet 0.25) and load weights.
    """
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
    """
    Ensure [x1,y1,x2,y2] stays within image bounds.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


# --------------------------
# Detector Wrapper
# --------------------------
class RetinaFaceDetector:
    """
    Lightweight RetinaFace detector (MobileNet 0.25 backbone).
    Given a BGR image, returns list of bounding boxes (x1,y1,x2,y2) and scores.
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
        """
        Args:
            img_bgr: HxWx3 (uint8) BGR image (cv2.imread).

        Returns:
            List[{"bbox": [x1,y1,x2,y2], "score": float}]
        """
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "Expected HxWx3 BGR image."
        img = np.float32(img_bgr.copy())
        im_height, im_width, _ = img.shape

        # Preprocess: subtract BGR mean as in repo
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)  # CHW
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Forward
        loc, conf, landms = self.net(img_t)  # [1,N,4], [1,N,2], [1,N,10]

        # Prior boxes
        priors = PriorBox(self.cfg, image_size=(im_height, im_width)).forward().to(self.device)
        boxes = decode(loc.squeeze(0), priors, self.cfg['variance'])
        boxes = boxes * torch.tensor([im_width, im_height, im_width, im_height], device=self.device)

        scores = conf.squeeze(0)[:, 1]

        # Filter by score
        inds = torch.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        if boxes.numel() == 0:
            return []

        # Keep top_k before NMS
        scores, order = scores.sort(descending=True)
        order = order[:self.top_k]
        boxes = boxes[order]

        # NMS
        dets = torch.cat([boxes, scores.unsqueeze(1)], dim=1).detach().cpu().numpy()  # Nx5

        if _USE_TV_NMS:
            # torchvision.ops.nms expects tensors [x1,y1,x2,y2], scores
            tb = torch.tensor(dets[:, :4])
            ts = torch.tensor(dets[:, 4])
            keep_idx = torch.ops.torchvision.nms(tb, ts, self.nms_thresh).numpy().tolist()
        else:
            keep_idx = nms_cpu(dets, self.nms_thresh)

        dets = dets[keep_idx]
        dets = dets[:self.keep_top_k, :]

        # Build result
        results = []
        for x1, y1, x2, y2, s in dets:
            box = clip_box_to_image(np.array([x1, y1, x2, y2]), im_width, im_height)
            results.append({"bbox": box.tolist(), "score": float(s)})
        return results


# --------------------------
# Lightweight Feature Extractor
# --------------------------
class MobileNetV2Head(nn.Module):
    """
    Lightweight embedding head using MobileNetV2 backbone (ImageNet pretrained).
    Outputs L2-normalized 128-D feature vectors. (Swap to MobileFaceNet/ArcFace later.)
    """

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
        x = F.normalize(x, p=2, dim=1)
        return x


class FaceFeatureExtractor:
    """
    Crops faces and extracts fixed-dim embeddings with a light model.
    """

    def __init__(self, device: str | torch.device = None, out_dim: int = 128, input_size: int = 224):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.model = MobileNetV2Head(out_dim=out_dim).to(self.device).eval()
        self.input_size = input_size

        self.tf = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_one(self, img_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Args:
            img_bgr: original BGR image
            bbox: [x1,y1,x2,y2]
        Returns:
            (D,) numpy array (float32) embedding
        """
        x1, y1, x2, y2 = bbox
        crop = img_bgr[y1:y2, x1:x2, :]  # BGR
        if crop.size == 0:
            return np.zeros(self.model.fc.out_features, dtype=np.float32)

        # Convert BGR->RGB for torchvision models
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = self.tf(crop_rgb).unsqueeze(0).to(self.device)  # 1x3xHxW
        feat = self.model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return feat


# --------------------------
# Rendering with temporary IDs
# --------------------------
def draw_detections_with_ids(
        img_bgr: np.ndarray,
        detections: List[Dict],
        seed: int | None = None
) -> np.ndarray:
    """
    Draw bboxes and temporary 4-digit IDs on the image.
    """
    if seed is not None:
        random.seed(seed)

    vis = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = random.randint(1000, 9999)  # temporary; to be replaced by tracker IDs later
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{track_id}"
        # Put text background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return vis


# --------------------------
# Pipeline helper
# --------------------------
def process_image(
        image_path: str,
        detector: RetinaFaceDetector,
        extractor: FaceFeatureExtractor,
        csv_out: str,
        image_out: str | None = None,
) -> List[Dict]:
    """
    1) Run detection -> list of boxes
    2) For each box, extract features
    3) Save CSV: x1,y1,x2,y2,feat_0,...,feat_(D-1)
    4) Optionally save annotated image with temporary IDs

    Returns detections (each with bbox + score)
    """
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Failed to read image: {image_path}"

    detections = detector.detect(img_bgr)

    # Build CSV rows
    rows = []
    D = extractor.model.fc.out_features
    for det in detections:
        bbox = det["bbox"]
        feat = extractor.extract_one(img_bgr, bbox)  # (D,)
        row = {
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            **{f"feat_{i}": float(feat[i]) for i in range(D)}
        }
        rows.append(row)

    # Save CSV
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    # Optional image with temp IDs
    if image_out is not None:
        vis = draw_detections_with_ids(img_bgr, detections)
        Path(image_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_out, vis)

    return detections


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    """
    Example:
      python3 face_pipeline_v1.py \
        --weights ./weights/mobilenet0.25_Final.pth \
        --image ./samples/sample.jpg \
        --csv_out ./outputs/test_features.csv \
        --img_out ./outputs/test_vis.jpg
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to mobilenet0.25_Final.pth")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_out", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--img_out", type=str, default=None, help="Optional path to save visualization")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if None)")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    args = parser.parse_args()

    detector = RetinaFaceDetector(
        weights_path=args.weights,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
    )
    extractor = FaceFeatureExtractor(device=args.device, out_dim=128, input_size=224)

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
