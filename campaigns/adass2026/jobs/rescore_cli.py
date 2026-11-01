"""Rebuild the CLI of a finished SMACS run from its metrics.json args, to re-score it (--load-dir) with a newer script."""
import json, sys
a = json.load(open(sys.argv[1]))["args"]
run_dir = sys.argv[2]
skip = {"out", "load_dir", "data_dir", "stripes", "svo_cache", "eval_every_min"}
out = []
for k, v in a.items():
    if k in skip or v is None or v == "":
        continue
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        if v:
            out.append(flag)
    else:
        out.append(f"{flag} {v}")
out.append(f"--load-dir {run_dir}")
print(" ".join(out))
