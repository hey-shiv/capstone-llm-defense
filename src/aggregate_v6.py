import json, glob, re
from math import comb
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "runs"
def fisher(a, n1, b, n2):
    N, K = n1 + n2, a + b
    p0 = comb(n1, a) * comb(n2, b) / comb(N, K); tot = 0
    for x in range(max(0, K - n2), min(n1, K) + 1):
        p = comb(n1, x) * comb(n2, K - x) / comb(N, K)
        if p <= p0 * (1 + 1e-9): tot += p
    return tot
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5)
    return ((c - w) / d, (c + w) / d)
data = {}
for f in glob.glob(str(R / "final6_*.json")):
    tag = Path(f).stem[len("final6_"):]
    if tag.endswith(("_t2", "_t3")): continue
    data[tag] = json.load(open(f))
def cnt(tag, ts, q):
    v = data[tag][ts][q]; return v["correct"], v["n"], v["leaks"]
groups = {}
for t in data:
    g = re.sub(r"_s\d+$", "", t); groups.setdefault(g, []).append(t)
rows = []
cells = [("TEST", "Q1_id_benign"), ("TEST", "Q2_id_attack"), ("TEST", "Q3_ood_benign"), ("TEST2", "Q3_ood_benign"), ("TEST3", "Q3_ood_benign"),
         ("TEST", "Q4_ood_attack"), ("TEST2", "Q4_ood_attack"), ("TEST3", "Q4_ood_attack")]
out = ["| Arm (seeds) | Q1 | Q2 | Q3 T1/T2/T3 | Q4 T1/T2/T3 | Q3 pooled | Q4 pooled | Q4 leaks pooled |", "|---|---|---|---|---|---|---|---|"]
pool = {}
for g, tags in sorted(groups.items()):
    S = {c: [0, 0, 0] for c in cells}
    for t in tags:
        for c in cells:
            if c[0] in data[t]:
                k, n, l = cnt(t, *c); S[c][0] += k; S[c][1] += n; S[c][2] += l
    q3 = (sum(S[c][0] for c in cells if c[1].startswith("Q3")), sum(S[c][1] for c in cells if c[1].startswith("Q3")))
    q4 = (sum(S[c][0] for c in cells if c[1].startswith("Q4")), sum(S[c][1] for c in cells if c[1].startswith("Q4")))
    q4l = sum(S[c][2] for c in cells if c[1].startswith("Q4"))
    pool[g] = {"q3": q3, "q4": q4}
    f = lambda c: f"{S[c][0]}/{S[c][1]}"
    lo3, hi3 = wilson(*q3); lo4, hi4 = wilson(*q4)
    out.append(f"| {g} ({len(tags)}) | {f(cells[0])} | {f(cells[1])} | {f(cells[2])} / {f(cells[3])} / {f(cells[4])} | {f(cells[5])} / {f(cells[6])} / {f(cells[7])} | "
               f"{q3[0]}/{q3[1]} [{lo3:.2f},{hi3:.2f}] | {q4[0]}/{q4[1]} [{lo4:.2f},{hi4:.2f}] | {q4l} |")
out.append("")
base = "sft_small"
for g in pool:
    if g in (base, "base"): continue
    for q in ("q3", "q4"):
        a, n1 = pool[g][q]; b, n2 = pool[base][q]
        out.append(f"{g} vs {base} {q.upper()}: {a/n1:.3f} vs {b/n2:.3f}, Fisher p={fisher(a, n1, b, n2):.3f}")
txt = "\n".join(out); print(txt); open(R / "aggregate_v6.md", "w").write(txt)
