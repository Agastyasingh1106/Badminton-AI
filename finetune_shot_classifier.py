"""
Adapts the shot classifier to broadcast footage with the hand-labelled hits in hit_labels.json.

The classifier trained on the group member's fixed-camera dataset doesn't carry over to broadcast
rallies (below always guessing the most common type), so this learns the 6 shot groups
(badminton/shots.py SHOT_GROUPS) from our own labelled hits. With few labels, it compares:

  pretrained   - the original classifier, its 18 types summed into groups
  embed_lr     - logistic regression on the pretrained network's pooled features
  finetune     - the whole pretrained network fine-tuned on group labels
  most_common  - always the most common group in the training rallies

Every rally is scored by a model that never saw it (leave-one-rally-out). The best method is
refit on all labelled hits and saved to runs/shot_model/broadcast_groups.pt only if it beats
most_common; the pipeline then shows shot groups from it.

  python finetune_shot_classifier.py
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from badminton.court import CourtMapper, load_corners
from badminton.shot_types import SERVE, embed, embed_lr_probs, hitter_poses, model_inputs, type_probs_to_groups
from badminton.shots import GROUP_IDS, GROUP_OF, MODEL_DIR, SHOT_GROUPS, SHOT_TYPES, ShotClassifier
from eval_hits import HIT_LABELS, VIDEO_DIR, tracked

PROJECT_ROOT = Path(__file__).resolve().parent
POSE_CACHE = PROJECT_ROOT / "runs" / "shot_eval" / "poses.pkl"
OUT_MODEL = MODEL_DIR / "broadcast_groups.pt"
OUT_METRICS = MODEL_DIR / "broadcast_metrics.json"


def labelled_hits():
    """[(video, frame, side, group id, pose sequence, first hit of the rally)] for every typed hit with a usable pose."""
    labels = {v: e for v, e in json.loads(HIT_LABELS.read_text(encoding="utf-8")).items() if e.get("hits_done")}
    cache = pickle.loads(POSE_CACHE.read_bytes()) if POSE_CACHE.is_file() else {}
    corners, out = load_corners(), []
    for video, entry in sorted(labels.items()):
        hits = [h for h in entry["hits"] if h.get("shot_type") is not None]
        first = min(h["frame"] for h in entry["hits"])
        todo = [h for h in hits if (video, h["frame"], h["side"]) not in cache]
        if todo:
            frames, _ = tracked(video, CourtMapper(corners[video]))
            seqs, ok = hitter_poses(VIDEO_DIR / f"{video}.mp4", frames, todo)
            for h, s, good in zip(todo, seqs, ok):
                cache[(video, h["frame"], h["side"])] = s if good else None
        for h in hits:
            seq = cache[(video, h["frame"], h["side"])]
            if seq is not None:
                out.append((video, h["frame"], h["side"], SHOT_GROUPS.index(GROUP_OF[SHOT_TYPES[h["shot_type"]]]), seq,
                            h["frame"] == first))
    POSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    POSE_CACHE.write_bytes(pickle.dumps(cache))
    return out


def pretrained_groups(clf, seqs):
    return type_probs_to_groups(clf.predict_batch(seqs))


def fit_embed_lr(clf, seqs, y):
    X = np.concatenate([embed(clf, model_inputs(seqs)), embed(clf, model_inputs(seqs, flip=True))])
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    return lr.fit(X, np.concatenate([y, y]))


def fit_finetune(seqs, y, epochs=40, seed=0):
    """Pretrained CNN, trained end to end on group labels (group prob = sum of its types' probs)."""
    torch.manual_seed(seed)
    clf = ShotClassifier()
    model = clf.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    x = torch.cat([model_inputs(seqs), model_inputs(seqs, flip=True)]).to(clf.device)
    t = torch.from_numpy(np.concatenate([y, y])).to(clf.device)
    group_of = torch.from_numpy(GROUP_IDS).to(clf.device)
    weights = torch.tensor([1.0 / max((y == g).sum(), 1) for g in range(len(SHOT_GROUPS))],
                           dtype=torch.float32, device=clf.device)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=clf.device)
        for i in range(0, len(x), 32):
            b = perm[i:i + 32]
            p = torch.softmax(model(x[b]), dim=1)
            g = torch.zeros(len(b), len(SHOT_GROUPS), device=clf.device).index_add_(1, group_of, p)
            loss = F.nll_loss(torch.log(g + 1e-8), t[b], weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return clf


def accuracy(probs, y):
    return float(np.mean(np.argmax(probs, axis=1) == y)) if len(y) else None


def main():
    parser = argparse.ArgumentParser(description="Adapt the shot classifier to broadcast footage.")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    data = labelled_hits()
    videos = sorted({d[0] for d in data})
    y_all = np.array([d[3] for d in data])
    print(f"{len(data)} labelled hits with a pose from {len(videos)} rallies; groups "
          f"{dict(Counter(SHOT_GROUPS[g] for g in y_all))}")
    if len(videos) < 3:
        raise SystemExit("Label at least 3 rallies with label_hits.py first.")

    base = ShotClassifier()
    preds = {m: np.zeros((len(data), len(SHOT_GROUPS))) for m in ("pretrained", "embed_lr", "finetune", "most_common")}
    for video in videos:
        test = np.array([d[0] == video for d in data])
        tr_seqs = [d[4] for d, t in zip(data, test) if not t]
        te_seqs = [d[4] for d, t in zip(data, test) if t]
        y_tr = y_all[~test]
        preds["pretrained"][test] = pretrained_groups(base, te_seqs)
        preds["embed_lr"][test] = embed_lr_probs(base, fit_embed_lr(base, tr_seqs, y_tr), te_seqs)
        preds["finetune"][test] = pretrained_groups(fit_finetune(tr_seqs, y_tr, args.epochs), te_seqs)
        preds["most_common"][test, Counter(y_tr.tolist()).most_common(1)[0][0]] = 1.0

    # The pipeline only lets a rally's first shot be a serve (classify_shots); score that rule too
    later = np.array([not d[5] for d in data])
    for m in ("embed_lr", "finetune"):
        ruled = preds[m].copy()
        ruled[later, SERVE] = 0.0
        preds[f"{m}+serve_rule"] = ruled
    scores = {m: accuracy(p, y_all) for m, p in preds.items()}
    print("\nShot-group accuracy on unseen rallies (leave-one-rally-out):")
    for m, s in scores.items():
        print(f"  {m:<20} {s:.0%}")

    best = max(("embed_lr", "finetune"), key=lambda m: scores[f"{m}+serve_rule"])
    metrics = {"n_hits": len(data), "n_rallies": len(videos), "evaluation": "leave-one-rally-out",
               "group_accuracy": scores, "chosen": None}
    if scores[f"{best}+serve_rule"] > scores["most_common"]:
        metrics["chosen"] = best
        seqs = [d[4] for d in data]
        if best == "embed_lr":
            torch.save({"kind": "embed_lr", "lr": pickle.dumps(fit_embed_lr(base, seqs, y_all))}, OUT_MODEL)
        else:
            torch.save({"kind": "finetune", "state_dict": fit_finetune(seqs, y_all, args.epochs).model.state_dict(),
                        "arch": base.arch}, OUT_MODEL)
        print(f"\n{best} beats always guessing the most common group; saved to {OUT_MODEL}")
    else:
        OUT_MODEL.unlink(missing_ok=True)
        print("\nNo method beats always guessing the most common group yet; nothing saved.")
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
