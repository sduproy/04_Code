#compute_log.py One row per run in 05_Outputs/compute_log.csv
#Usage:  T0 = time.perf_counter()  at the top of the script, then
#        log_compute("my_script.py", T0, used_gpu=True/False)  at the end.
#Python 3.7-safe, so the SLS env scorers can import it too.
import csv, datetime, pathlib, platform, time

LOG = pathlib.Path(__file__).resolve().parent.parent / "05_Outputs" / "compute_log.csv"
FIELDS = ["timestamp", "script", "wall_hours", "gpu_hours", "gpu_device", "host"]

def log_compute(script: str, t0: float, used_gpu: bool = False,
                path: str = None) -> dict:
    hrs = (time.perf_counter() - t0) / 3600
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        gpu = "none"
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "script": script, "wall_hours": round(hrs, 4),
           "gpu_hours": round(hrs, 4) if used_gpu else 0.0,
           "gpu_device": gpu, "host": platform.node()}
    path = pathlib.Path(path or LOG)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)
    print(f"compute: {row['wall_hours']} h wall, {row['gpu_hours']} GPU-h ({gpu}) -> {path.name}")
    return row
