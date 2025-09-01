
Component 1 (face detector and feature extractor):  [Pipeline — Detect → Align → Embed]\
A compact, production-ready pipeline that runs **RetinaFace (MobileNet-0.25)** for face detection, performs **ArcFace-style \(5\)-point alignment** to \(112\times112\), and extracts **L2-normalized embeddings** from a pluggable head: lightweight **MobileNetV2** (\(128\)-D) or **InsightFace** recognizers (e.g., `buffalo_l`). Outputs a tidy CSV with boxes and features, plus an optional annotated image.

## Highlights
- **All-in-one CLI**: Detect → Align → Embed in one pass.  
- **Fast & lean**: Tunable `--conf` and `--nms`; uses CPU/GPU via `--device`.  
- **Stable IDs**: Alignment before embedding reduces pose/illumination drift.  
- **Drop-in features**: For retrieval, clustering, de-dup, tracking, or re-ID bootstrapping.

## Quickstart
```bash
# Default: MobileNetV2 embeddings (PyTorch)
python3 face_pipeline.py \
  --weights ./weights/mobilenet0.25_Final.pth \
  --image ./samples/sample.jpg \
  --csv_out ./outputs/features.csv \
  --img_out ./outputs/vis.jpg

# ArcFace embeddings via InsightFace (recommended for recognition)
python3 face_pipeline.py \
  --weights ./weights/mobilenet0.25_Final.pth \
  --image ./samples/sample.jpg \
  --csv_out ./outputs/features.csv \
  --img_out ./outputs/vis.jpg \
  --backend insightface --insightface_model buffalo_l
```
Output: CSV with x1,y1,x2,y2,feat_0…feat_(D-1) and an optional visualization.

Refs for the tools mentioned: biubug6’s RetinaFace (MobileNet-0.25) and InsightFace packs (`buffalo_l`, etc.), and \
Retinaface used: (https://github.com/biubug6/Pytorch_Retinaface.git) \
ArcFace paper: (https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)
