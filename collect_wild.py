#collect_wild.py Register-driven, evidence-first collection
import pathlib, subprocess
import pandas as pd
from config import CFG
from registry import new_row, sha256

def fetch(url: str, out_dir: str) -> str:
    """One clip, capped at 720p, only for sources whose terms allow"""
    out = str(pathlib.Path(out_dir) / "%(id)s.%(ext)s")
    subprocess.run(["yt-dlp",
                    "-f", "bv*[height<=720]+ba/b[height<=720]",
                    "-o", out, url], check=True)
    files = sorted(pathlib.Path(out_dir).glob("*"),
                   key=lambda q: q.stat().st_mtime)
    return str(files[-1])

def register_fake(clip_id, path, origin_url, evidence_url,
                  identity, source, licence) -> dict:
    assert evidence_url, "a fake without evidence is set aside"
    return new_row(clip_id=clip_id, label="fake",
                   source_dataset=source, origin_url=origin_url,
                   licence=licence, sha256=sha256(path),
                   identity=identity, generator="unknown_wild",
                   evidence_url=evidence_url,
                   date_documented=pd.Timestamp.today().date()
                   .isoformat())

def register_real(clip_id, path, origin_url, identity,
                  source, licence) -> dict:
    return new_row(clip_id=clip_id, label="real",
                   source_dataset=source, origin_url=origin_url,
                   licence=licence, sha256=sha256(path),
                   identity=identity, generator="none",
                   evidence_url="authentic: " + origin_url,
                   date_documented=pd.Timestamp.today().date()
                   .isoformat())

#Matching rules for the authentic side, in order:
#   1. Same persona or brand as the fake, different footage.
#   2. Same content type (endorsement, product pitch) matched on 
#   duration and production style
#Record which rule produced each match in the collection log.