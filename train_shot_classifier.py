"""
Trains the shot-type classifier (badminton/shots.py) on the group member's pose keypoints in
dataset/shot_keypoints.csv (7,818 clips, 18 shot types, YOLOv8n-pose on fixed-camera recordings).

The clips are split by recording session (the timestamp at the start of video_id), so no session is in
both train and test: our broadcast footage is a different setup, so scores on unseen sessions are the
ones that matter. Compares the group member's checkpoint as it is, a retrained LSTM, a GRU and a 1D-CNN
(class-weighted loss, mirror / time-shift / scale augmentation), picks the architecture by validation
macro-F1 and reports test top-1, top-3 and macro-F1 for the 18 types and the 6 coarse groups.
"""

import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from badminton.court import PROJECT_ROOT
from badminton.shots import (MODEL_DIR, MODEL_PATH, SEQ_LEN, SHOT_GROUPS, SHOT_TYPES, GROUP_IDS, build_model,
                             clip_features, fit_length, flip_clip, normalise_clip, normalise_root)

CSV_PATH = PROJECT_ROOT / "dataset" / "shot_keypoints.csv"
THEIR_CHECKPOINT = MODEL_DIR / "their_lstm.pth"
ARCHS = ("lstm", "gru", "cnn")
N_CLASSES = len(SHOT_TYPES)


def load_clips(path=CSV_PATH):
    """[(video_id, (T, 17, 3) pixel keypoints, label id)] in file order of video_id."""
    import pandas as pd
    print(f"Reading {path} ...")
    df = pd.read_csv(path).sort_values(["video_id", "frame_idx"], kind="stable")
    kp_cols = [f"kp_{j}_{c}" for j in range(17) for c in ("x", "y", "conf")]
    kpts = df[kp_cols].to_numpy(np.float32).reshape(-1, 17, 3)
    ids, starts = np.unique(df["video_id"].to_numpy(), return_index=True)
    labels = df["label_id"].to_numpy()[starts]
    ends = list(starts[1:]) + [len(df)]
    return [(vid, kpts[s:e], int(lab)) for vid, s, e, lab in zip(ids, starts, ends, labels)]


def session_of(video_id: str) -> str:
    return video_id[:19]   # e.g. 2022-08-30_18-00-09


def split_by_session(clips, val_frac=0.15, test_frac=0.15, seed=42, tries=500):
    """
    Assigns whole sessions to train/val/test. Sessions differ a lot in size (3 to 266 clips), so it
    tries many random assignments and keeps the one whose val/test sizes and class mix are closest
    to the target (every class must appear at least 5 times in val and in test).
    """
    by_session = Counter(session_of(v) for v, _, _ in clips)
    labels_of = {}
    for v, _, lab in clips:
        labels_of.setdefault(session_of(v), Counter())[lab] += 1
    sessions = sorted(by_session)
    total = np.bincount([lab for _, _, lab in clips], minlength=N_CLASSES).astype(float)
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(tries):
        order = rng.permutation(sessions)
        part, filled = {}, {"val": 0, "test": 0}
        for s in order:
            for name, frac in (("test", test_frac), ("val", val_frac)):
                if filled[name] < frac * len(clips):
                    part[s] = name
                    filled[name] += by_session[s]
                    break
            else:
                part[s] = "train"
        cost = 0.0
        for name, frac in (("val", val_frac), ("test", test_frac)):
            counts = np.zeros(N_CLASSES)
            for s, p in part.items():
                if p == name:
                    for lab, n in labels_of[s].items():
                        counts[lab] += n
            if counts.min() < 5:
                cost = np.inf
                break
            cost += abs(counts.sum() / len(clips) - frac) * 10 + np.abs(counts / counts.sum() - total / total.sum()).sum()
        if cost < best_cost:
            best, best_cost = part, cost
    if best is None:
        raise SystemExit("Could not find a session split with every class in val and test.")
    return best


class Normalised:
    """Clips normalised once (badminton/shots.py), cropped / mirrored / scaled per epoch."""

    def __init__(self, clips):
        self.clips = [(normalise_clip(k), normalise_root(k)) for _, k, _ in clips]
        self.y = np.array([lab for _, _, lab in clips])

    def features(self, rng=None, flip=False):
        out = []
        for clip, root in self.clips:
            offset, mirror, scale = 0, flip, 1.0
            if rng is not None:
                offset, mirror, scale = int(rng.integers(-4, 5)), bool(rng.random() < 0.5), rng.uniform(0.9, 1.1)
            c, r = fit_length(clip, offset=offset), fit_length(root, offset=offset)
            if mirror:
                c, r = flip_clip(c, r)
            if scale != 1.0:
                c = c.copy()
                c[..., :2] *= scale
                r = r * scale
            out.append(clip_features(c, r))
        return np.stack(out)


def predict_probs(model, x_plain, x_flip, device, batch=512):
    """Softmax averaged over the clip and its mirror image (same as ShotClassifier.predict)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x in (x_plain, x_flip):
            parts = [torch.softmax(model(torch.from_numpy(x[i:i + batch]).to(device)), 1).cpu().numpy()
                     for i in range(0, len(x), batch)]
            probs.append(np.concatenate(parts))
    return (probs[0] + probs[1]) / 2


def scores(probs: np.ndarray, y: np.ndarray) -> dict:
    top = np.argsort(-probs, axis=1)
    pred = top[:, 0]
    gp, gy = GROUP_IDS[pred], GROUP_IDS[y]   # group of the predicted shot type, so it never contradicts it
    return {
        "top1": float(np.mean(pred == y)),
        "top3": float(np.mean([t in row for t, row in zip(y, top[:, :3])])),
        "macro_f1": float(f1_score(y, pred, labels=range(N_CLASSES), average="macro", zero_division=0)),
        "group_top1": float(np.mean(gp == gy)),
        "group_macro_f1": float(f1_score(gy, gp, labels=range(len(SHOT_GROUPS)),
                                         average="macro", zero_division=0)),
        "classes_predicted": int(len(set(pred.tolist()))),
    }


def train_one(arch, seed, train, val, device, epochs, batch=64):
    """Trains one model; returns it with the weights of its best validation macro-F1 epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(arch).to(device)
    counts = np.bincount(train.y, minlength=N_CLASSES)
    weights = torch.tensor(len(train.y) / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    lr = 1e-3 if arch == "lstm" else 2e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=int(np.ceil(len(train.y) / batch)))
    x_val, x_val_flip = val.features(), val.features(flip=True)
    y_train = torch.from_numpy(train.y).to(device)
    best_f1, best_state, best_epoch = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        x = torch.from_numpy(train.features(rng)).to(device)
        for idx in torch.randperm(len(x), device=device).split(batch):
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        f1 = scores(predict_probs(model, x_val, x_val_flip, device), val.y)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch


class TheirLSTM(nn.Module):
    """The group member's model (train_lstm.py), to load best_badminton_lstm.pth unchanged."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(51, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1, self.relu, self.dropout, self.fc2 = nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 18)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(out[:, -1]))))


def their_checkpoint_probs(clips, device):
    """Their preprocessing: raw pixel keypoints, central 30 frames, zero padding at the end."""
    if not THEIR_CHECKPOINT.is_file():
        return None
    model = TheirLSTM().to(device).eval()
    model.load_state_dict(torch.load(THEIR_CHECKPOINT, map_location=device))
    x = []
    for _, k, _ in clips:
        k = k.reshape(len(k), 51)
        if len(k) >= SEQ_LEN:
            start = (len(k) - SEQ_LEN) // 2
            x.append(k[start:start + SEQ_LEN])
        else:
            x.append(np.vstack([k, np.zeros((SEQ_LEN - len(k), 51), np.float32)]))
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(np.stack(x)).to(device)), 1).cpu().numpy()


def majority_probs(train_y, n):
    """Always the most common training class; the rest ranked by training frequency (for top-3)."""
    freq = np.bincount(train_y, minlength=N_CLASSES).astype(float)
    probs = freq / freq.sum()
    return np.tile(probs, (n, 1))


def majority_group_scores(train_y, y):
    group = np.bincount(GROUP_IDS[train_y], minlength=len(SHOT_GROUPS)).argmax()
    gy = GROUP_IDS[y]
    return float(np.mean(gy == group)), float(f1_score(gy, np.full(len(gy), group), labels=range(len(SHOT_GROUPS)),
                                                       average="macro", zero_division=0))


def fmt(m):
    return (f"top-1 {m['top1']:6.1%}  top-3 {m['top3']:6.1%}  macro-F1 {m['macro_f1']:.3f}  |  "
            f"groups top-1 {m['group_top1']:6.1%}  macro-F1 {m['group_macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the shot-type classifier.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3, help="Training runs per architecture.")
    parser.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    clips = load_clips()
    part = split_by_session(clips)
    sets = {name: [c for c in clips if part[session_of(c[0])] == name] for name in ("train", "val", "test")}
    for name, cs in sets.items():
        n_sessions = len({session_of(v) for v, _, _ in cs})
        print(f"{name}: {len(cs)} clips from {n_sessions} sessions")
    t0 = time.time()
    data = {name: Normalised(cs) for name, cs in sets.items()}
    print(f"Normalised clips in {time.time() - t0:.0f}s")
    x_test, x_test_flip = data["test"].features(), data["test"].features(flip=True)
    y_test = data["test"].y

    results = {}
    maj = majority_probs(data["train"].y, len(y_test))
    results["majority_class"] = scores(maj, y_test)
    results["majority_class"]["group_top1"], results["majority_class"]["group_macro_f1"] = \
        majority_group_scores(data["train"].y, y_test)
    theirs = their_checkpoint_probs(sets["test"], device)
    if theirs is not None:
        # Their random clip split put most of these test sessions' clips in their training set, so this is optimistic
        results["their_checkpoint"] = scores(theirs, y_test)

    runs, best_models = {}, {}
    for arch in args.archs:
        runs[arch] = []
        for seed in range(args.seeds):
            t0 = time.time()
            model, val_f1, epoch = train_one(arch, seed, data["train"], data["val"], device, args.epochs)
            test = scores(predict_probs(model, x_test, x_test_flip, device), y_test)
            runs[arch].append({"seed": seed, "val_macro_f1": val_f1, "best_epoch": epoch, "test": test})
            print(f"  {arch} seed {seed}: val macro-F1 {val_f1:.3f} (epoch {epoch})  test {fmt(test)}  "
                  f"[{time.time() - t0:.0f}s]")
            if arch not in best_models or val_f1 > best_models[arch][0]:
                best_models[arch] = (val_f1, seed, model)
        mean_val = np.mean([r["val_macro_f1"] for r in runs[arch]])
        results[arch] = {k: float(np.mean([r["test"][k] for r in runs[arch]])) for k in runs[arch][0]["test"]}
        results[arch]["test_std"] = {k: float(np.std([r["test"][k] for r in runs[arch]]))
                                     for k in ("top1", "top3", "macro_f1", "group_top1", "group_macro_f1")}
        results[arch]["val_macro_f1"] = float(mean_val)

    # Architecture by mean validation macro-F1 over seeds; saved weights = its best-validation seed
    chosen = max(args.archs, key=lambda a: results[a]["val_macro_f1"])
    _, seed, model = best_models[chosen]
    probs = predict_probs(model, x_test, x_test_flip, device)
    saved = scores(probs, y_test)
    pred = probs.argmax(1)
    per_class_f1 = f1_score(y_test, pred, labels=range(N_CLASSES), average=None, zero_division=0)

    print(f"\nTest set: {len(y_test)} clips from sessions never seen in training "
          f"(scores for trained models = mean of {args.seeds} seeds)")
    for name in ("majority_class", "their_checkpoint", *args.archs):
        if name in results:
            print(f"  {name:<17} {fmt(results[name])}  ({results[name]['classes_predicted']} classes predicted)")
    print(f"  saved: {chosen} seed {seed}  {fmt(saved)}")
    print("  Per-class test F1:", ", ".join(f"{n} {f:.2f}" for n, f in zip(SHOT_TYPES, per_class_f1)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": chosen, "state_dict": model.state_dict(), "classes": SHOT_TYPES, "groups": SHOT_GROUPS,
                "seq_len": SEQ_LEN, "seed": seed}, MODEL_PATH)
    metrics = {
        "evaluation": "train/val/test split by recording session (no session in two splits); "
                      "test scores of trained models are the mean over seeds",
        "n_clips": {k: len(v) for k, v in sets.items()},
        "sessions": {k: sorted({session_of(v) for v, _, _ in cs}) for k, cs in sets.items()},
        "chosen_arch": chosen, "saved_seed": seed, "saved_model_test": saved,
        "results": results, "runs": runs,
        "per_class_test_f1": dict(zip(SHOT_TYPES, map(float, per_class_f1))),
        "test_confusion": confusion_matrix(y_test, pred, labels=range(N_CLASSES)).tolist(),
        "note": "their_checkpoint was trained on a random clip split of the whole CSV, so most of these "
                "test clips were in its training data; its score is optimistic.",
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
