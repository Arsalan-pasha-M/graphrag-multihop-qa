"""Download the HotpotQA dev (distractor) set (~45 MB) to data/."""
import pathlib, requests
URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "hotpot_dev_distractor_v1.json"
out.parent.mkdir(exist_ok=True)
if out.exists():
    print("Already downloaded:", out)
else:
    print("Downloading", URL)
    r = requests.get(URL, timeout=120); r.raise_for_status()
    out.write_bytes(r.content); print("Saved", out)
