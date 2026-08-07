# frames_faces_audio.py Deterministic, identical for every dataset
import pathlib
import cv2
import numpy as np
import librosa
from facenet_pytorch import MTCNN
from config import CFG


mtcnn = MTCNN(image_size=CFG.face_size, margin=20,
              post_process=False, keep_all=False)

def frames(video_path: str, out_dir: str) -> list:
    vid = pathlib.Path(video_path).stem
    out = pathlib.Path(out_dir) / vid
    out.mkdir(parents = True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / CFG.frame_fps))
    paths, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            fp = out / f"{vid}_{i:06d}.jpg"
            cv2.imwrite(str(fp), frame,
                        [cv2.IMWRITE_JPEG_QUALITY], 92)
            paths.append(str(fp))
        i += 1
    cap.release()
    return paths

def visual_stats(frame_paths: list) -> dict:
    sharp, boxes = [], []
    for p in frame_paths:
        im = cv2.imread(p)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F). var())
        b, _ = mtcnn.detect(im[..., ::-1])
        if b is not None and len(b):
            x0, y0, x1, y1 = b[0]
            boxes.append(((x0 + x1) / 2 (y0 + y1) / 2))
    jitter = (float(np.std(boxes, axis = 0).mean())
              if len(boxes) > 2 else 0.0)
    return {"v_sharpness_mean": float(np.mean(sharp)),
            "v_sharpness_std": float(np.std(sharp)),
            "v_face_rate": len(boxes) / max(len(frame_paths), 1),
            "v_facebox_jitter": jitter}

def audio_feats(video_path: str) -> dict:
    y, sr = librosa.load(video_path, sr=CFG.audio_sr, mono=True)
    if len(y) < sr:
        return {"a_ok": 0}
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG.n_mfcc)
    d = librosa.feature.delta(m)
    out = {"a_ok": 1}
    for name, mat in (("m", m), ("d", d)):
        for i in range(mat.shape[0]):
            out[f"a_{name}{i}_mean"] = float(mat[i].mean())
            out[f"a_{name}{i}_std"] = float(mat[i].std())
    return out