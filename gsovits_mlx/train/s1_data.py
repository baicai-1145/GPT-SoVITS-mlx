"""s1 (AR/GPT) training dataset + bucket sampler — official-semantics port.

Port of GPT-SoVITS-cuda_graph_accel_v5/GPT_SoVITS/AR/data/{dataset,
bucket_sampler}.py (Apache-2.0, upstream SoundStorm/VALL-E). The dataset
reads the s1 pipeline artifacts (2-name2text.txt, 3-bert/*.npy, 6-name2semantic.tsv)
produced by the official 1-prepare step (task-2 pipeline), applies the
EXACT official filters, and collates right-padded batches. The sampler
sorts by seconds, buckets, shuffles within buckets per epoch, and regroups
— the length-bucketing that makes the official forward_old single-slice
logits valid (rows in a batch have near-equal lengths).

Official quirks kept faithfully (see inline comments):

* phoneme-file line filter: keep only lines with exactly 4 tab-separated
  fields (the 4th field "text" is parsed but unused downstream);
* min_num=100 duplication: replicate the WHOLE kept list
  max(2, int(100/len)) times when fewer than 100 samples survive;
* semantic length filter compares len(semantic_ids) > max_sec*hz (hz=25,
  max_sec from the target version's s1 config: 54 for v1, 57 v2-family);
* missing BERT features -> zeros (never skip; the official assert on
  shape mismatch is a hard error, kept as ValueError);
* sampler pad-to-total via index repetition (drop_last=False); batch
  order shuffled after grouping.

BERT storage: 3-bert/*.npy float16 (1024, T) produced by our pipeline;
loaded per item, zero-padded to (B, 1024, max_phoneme_len) in collate.
cleaned_text_to_sequence comes from the vendored official text package
(gsovits_mlx.text.vendored_cpufront) with the version symbol table chosen
by the explicit ``version`` argument (v1: 512 symbols, v2: 732) — the
official module reads os.environ["version"] only as its fallback default.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["Text2SemanticDataset", "BucketBatchSampler", "collate",
           "pad_y_eos", "batch_sequences"]

HZ = 25  # official env default "25hz" (AR/data/dataset.py)

# gsovits_mlx/train/s1_data.py is five dirnames from the MAIN checkout
# root when installed as the repo package; in worktrees the tree is
# <main>/.pi/.../worktrees/<name>/gsovits_mlx/train/ — walk up until a
# models_local dir is found (main checkout owns the weights).
def _default_models_root() -> str:
    if os.environ.get("GSOVITS_MODELS_ROOT"):
        return os.environ["GSOVITS_MODELS_ROOT"]
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "models_local")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.join("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx",
                        "models_local")


DEFAULT_MODELS_ROOT = _default_models_root()


def bootstrap_frontend(cpufast_repo: str | None = None,
                      models_root: str | None = None) -> str:
    """Official front-end bootstrap before importing the vendored 'text'
    package: chdir to the CPUFast repo (relative G2PW/japanese scratch
    paths) + set bert_path (G2PW tokenizer source). Mirrors
    gsovits_mlx.text.preproc.bootstrap with the repo models_local default.

    Also pins GSOVITS_G2PW_SAFETENSORS/GSOVITS_MODELS_ROOT when a weights
    dir is resolvable: gsovits_mlx.text.g2pw_mlx freezes its
    DEFAULT_MODELS_ROOT at import time (``os.environ.get`` at module
    scope) and worktrees don't carry models_local — without this pin a
    same-process earlier import of g2pw_mlx leaves the frozen worktree
    path and the G2PW weights are never found.

    Returns the CPUFast repo path (cwd is LEFT there — callers holding
    relative paths must absolutize first; official parity requirement).
    """
    from gsovits_mlx.text import vendored_cpufront as vcf

    repo = vcf.cpufast_repo(cpufast_repo)
    if os.getcwd() != repo:
        os.chdir(repo)
    root = models_root or DEFAULT_MODELS_ROOT
    # propagate to the vendored modules (g2pw_mlx resolves weights via
    # GSOVITS_MODELS_ROOT; worktrees don't carry models_local themselves)
    os.environ.setdefault("GSOVITS_MODELS_ROOT", root)
    g2pw_w = os.path.join(root, "g2pw", "g2pw.safetensors")
    if os.path.exists(g2pw_w):
        os.environ.setdefault("GSOVITS_G2PW_SAFETENSORS", g2pw_w)
    if not os.environ.get("bert_path"):
        os.environ["bert_path"] = os.path.join(root, "bert")
    return repo


def _read_phoneme_file(path: str) -> dict:
    """2-name2text.txt -> {name: [phoneme, word2ph, text]} (4-field lines)."""
    phoneme_data: dict = {}
    with open(path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    for line in lines:
        tmp = line.split("\t")
        if len(tmp) != 4:
            continue
        phoneme_data[tmp[0]] = [tmp[1], tmp[2], tmp[3]]
    return phoneme_data


def _load_bert_feature(bert_dir: str, item_name: str) -> np.ndarray | None:
    """3-bert/<name>.npy (our pipeline) -> (1024, T) fp16.

    The official loader reads torch .pt files; our pipeline (task-2) writes
    .npy so the trainer stays torch-free. A stray official .pt raises a
    clear error instead of a corrupt unpickle. Missing file -> None (zeros
    downstream, official behavior).
    """
    path_npy = os.path.join(bert_dir, item_name + ".npy")
    if os.path.exists(path_npy):
        return np.load(path_npy)
    path_pt = os.path.join(bert_dir, item_name + ".pt")
    if os.path.exists(path_pt):
        raise ValueError(
            f"official torch .pt bert feature found but torch-free trainer "
            f"cannot read it: {path_pt} (re-run the pipeline to emit .npy)")
    return None


def pad_y_eos(codes: np.ndarray, y_mask_int: np.ndarray, eos: int = 1024):
    """Official pad_y_eos (t2s_model.py): shifted targets + EOS bookkeeping.

    codes (B, Y) int, y_mask_int (B, Y) int in {0,1} (1 = pad position).
    Returns (y, targets):
        targets = pad(codes, (0,1), 0) + eos * pad(y_mask_int, (0,1), 1)
        y, targets = targets[:, :-1], targets
    i.e. the decoder input y is codes shifted right by one (first input is
    the token codes[0] at position 0... see forward notes: y row starts
    with its first real code; EOS marks every pad slot AND the final
    target), and targets adds an EOS at the end of every row.
    """
    B, Y = codes.shape
    tgt = np.zeros((B, Y + 1), dtype=np.int64)
    tgt[:, :Y] = codes + eos * y_mask_int
    tgt[:, Y] = eos
    return tgt[:, :-1].copy(), tgt


class Text2SemanticDataset:
    """Official Text2SemanticDataset on numpy (no pandas/torch)."""

    def __init__(
        self,
        phoneme_path: str,
        semantic_path: str,
        version: str = "v2",
        max_sample: int | None = None,
        max_sec: int = 100,
        pad_val: int = 1024,
        min_ps_ratio: float = 3,
        max_ps_ratio: float = 25,
        hz: int = HZ,
        cpufast_repo: str | None = None,
        models_root: str | None = None,
    ):
        self.version = version
        self.path2 = phoneme_path
        self.exp_dir = os.path.dirname(phoneme_path)
        self.path3 = os.path.join(self.exp_dir, "3-bert")
        self.path6 = semantic_path
        if not os.path.exists(self.path2):
            raise FileNotFoundError(f"Phoneme data file not found: {self.path2}")
        if not os.path.exists(self.path6):
            raise FileNotFoundError(f"Semantic data file not found: {self.path6}")
        self.PAD = pad_val
        self.hz = hz
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        # semantic tsv: OFFICIAL FORMAT has a pandas-read-csv header line
        # ("item_name\tsemantic_audio") then name\t"id id ..." rows — the
        # official dataset uses pandas.read_csv(delimiter="\t") which CONSUMES
        # the header; headerless variant (official dump_mix) also accepted.
        semantic_data: list = []
        with open(semantic_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        if lines and lines[0].startswith("item_name"):
            lines = lines[1:]
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            semantic_data.append((parts[0], parts[1]))
        if max_sample is not None:
            semantic_data = semantic_data[:max_sample]

        phoneme_data = _read_phoneme_file(phoneme_path)

        # cleaned_text_to_sequence via the vendored official text package.
        # The official dataset reads os.environ["version"] at module import
        # into a default; we pass the version explicitly per call, but also
        # set the env var for parity with any vendored module that reads it.
        # bootstrap (chdir + bert_path) MUST precede load_cpufront — the
        # G2PW converter resolves model_dir/tokenizer relative to cwd.
        os.environ["version"] = version
        from gsovits_mlx.text import vendored_cpufront as vcf

        bootstrap_frontend(cpufast_repo, models_root)
        vcf.load_cpufront()
        import text as text_pkg  # the vendored official package

        self.semantic_phoneme: list = []
        self.item_names: list = []
        self.stats = dict(num_not_in=0, num_deleted_bigger=0,
                          num_deleted_ps=0, num_duplicated=0)
        num_not_in = num_deleted_bigger = num_deleted_ps = 0
        for item_name, semantic_str in semantic_data:
            entry = phoneme_data.get(item_name)
            if entry is None:
                num_not_in += 1
                continue
            phoneme, _word2ph, _text = entry
            semantic_ids = [int(tok) for tok in semantic_str.split(" ")]
            if len(semantic_ids) > self.max_sec * self.hz:
                num_deleted_bigger += 1
                continue
            phoneme_list = phoneme.split(" ")
            try:
                phoneme_ids = text_pkg.cleaned_text_to_sequence(
                    phoneme_list, self.version)
            except Exception:
                num_not_in += 1
                continue
            if len(phoneme_ids) > self.max_sec * self.hz / 2.5:
                num_deleted_ps += 1
                continue
            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                num_deleted_ps += 1
                continue
            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            self.item_names.append(item_name)

        # min_num duplication rule (official): replicate the whole kept list
        leng = len(self.semantic_phoneme)
        if 0 < leng < 100:
            reps = max(2, int(100 / leng))
            self.semantic_phoneme = self.semantic_phoneme * reps
            self.item_names = self.item_names * reps
            self.stats["num_duplicated"] = reps
        self.stats.update(num_not_in=num_not_in,
                          num_deleted_bigger=num_deleted_bigger,
                          num_deleted_ps=num_deleted_ps)
        print(f"dataset.__len__(): {len(self)}  {self.stats}")

    # -- dataset protocol ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def get_sample_length(self, idx: int) -> float:
        semantic_ids = self.semantic_phoneme[idx][0]
        return 1.0 * len(semantic_ids) / self.hz

    def __getitem__(self, idx: int) -> dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name = self.item_names[idx]
        bert = _load_bert_feature(self.path3, item_name)
        if bert is not None and bert.shape[-1] != len(phoneme_ids):
            raise ValueError(
                f"BERT feature dimension ({bert.shape[-1]}) of '{item_name}' "
                f"does not match phoneme length ({len(phoneme_ids)})")
        return {
            "idx": idx,
            "phoneme_ids": np.asarray(phoneme_ids, dtype=np.int64),
            "phoneme_ids_len": len(phoneme_ids),
            "semantic_ids": np.asarray(semantic_ids, dtype=np.int64),
            "semantic_ids_len": len(semantic_ids),
            "bert_feature": bert,
        }

    # -- collate ---------------------------------------------------------------
    def collate(self, examples: list[dict]) -> dict:
        return collate(examples, self.PAD)


def collate(examples: list[dict], pad_val: int = 1024) -> dict:
    """Right-pad phones (0) / semantics (pad_val); bert zero-padded.

    Returns dict of numpy arrays:
      ids (list[int]), phoneme_ids (B,X) i64, phoneme_ids_len (B,) i64,
      semantic_ids (B,Y) i64, semantic_ids_len (B,) i64,
      bert_feature (B,1024,Xmax) f16.
    """
    B = len(examples)
    phoneme_lens = [len(e["phoneme_ids"]) for e in examples]
    semantic_lens = [len(e["semantic_ids"]) for e in examples]
    X = max(phoneme_lens)
    Y = max(semantic_lens)
    phoneme_ids = np.zeros((B, X), dtype=np.int64)
    semantic_ids = np.full((B, Y), pad_val, dtype=np.int64)
    phoneme_ids_len = np.asarray(phoneme_lens, dtype=np.int64)
    semantic_ids_len = np.asarray(semantic_lens, dtype=np.int64)
    bert_padded = np.zeros((B, 1024, X), dtype=np.float16)
    for i, e in enumerate(examples):
        pl = phoneme_lens[i]
        sl = semantic_lens[i]
        phoneme_ids[i, :pl] = e["phoneme_ids"]
        semantic_ids[i, :sl] = e["semantic_ids"]
        bert = e["bert_feature"]
        if bert is not None:
            bert_padded[i, :, :pl] = bert[:, :pl]
    return {
        "ids": [e["idx"] for e in examples],
        "phoneme_ids": phoneme_ids,
        "phoneme_ids_len": phoneme_ids_len,
        "semantic_ids": semantic_ids,
        "semantic_ids_len": semantic_ids_len,
        "bert_feature": bert_padded,
    }


# ---------------------------------------------------------------------------
# Bucket sampler (official DistributedBucketSampler, num_replicas=1)
# ---------------------------------------------------------------------------

class BucketBatchSampler:
    """Sort by seconds -> 2.0s buckets -> per-epoch in-bucket shuffle ->
    chain -> regroup by batch_size -> batch-order shuffle -> pad to total.

    Deterministic given (seed, epoch): python random.Random(seed+epoch)
    drives every shuffle (the official global-random path with
    num_replicas=1; the official torch.Generator instance is never used).
    NOTE the official shuffles each bucket copy with the GLOBAL random
    module seeded once per __iter__ — same stream as Random(seed+epoch).
    """

    def __init__(self, dataset: Text2SemanticDataset, batch_size: int = 32,
                 shuffle: bool = True, seed: int = 0,
                 bucket_width: float = 2.0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = 0
        self.id_with_length = [(i, dataset.get_sample_length(i))
                               for i in range(len(dataset))]
        self.id_with_length.sort(key=lambda x: x[1])
        self.id_buckets = self.make_buckets(bucket_width)

    def make_buckets(self, bucket_width: float = 2.0) -> list[list[int]]:
        buckets: list[list[int]] = []
        cur: list[int] = []
        max_sec = bucket_width
        for idx, sec in self.id_with_length:
            if sec < max_sec:
                cur.append(idx)
            else:
                buckets.append(cur)
                cur = [idx]
                max_sec += bucket_width
        if len(cur) > 0:
            buckets.append(cur)
        return buckets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batch_indices(self, epoch: int | None = None) -> list[list[int]]:
        """Materialized list of batches (each a list of dataset indices)."""
        if not self.shuffle:
            all_idx = list(range(len(self.dataset)))
            return [all_idx[i:i + self.batch_size]
                    for i in range(0, len(all_idx), self.batch_size)]
        epoch = self.epoch if epoch is None else epoch
        rng = random.Random(self.seed + epoch)
        shuffled_bucket = []
        for buc in self.id_buckets:
            buc_copy = list(buc)
            rng.shuffle(buc_copy)
            shuffled_bucket.append(buc_copy)
        grouped = self.batch_size  # * num_replicas (1)
        chained = [i for buc in shuffled_bucket for i in buc]
        n_batch = (len(chained) + grouped - 1) // grouped
        batches = [chained[b * grouped:(b + 1) * grouped]
                   for b in range(n_batch)]
        rng.shuffle(batches)
        # official drop_last=False: pad the index stream to total via
        # repetition. With num_replicas=1 total == len(dataset) may exceed
        # the shuffled stream only when batch_size * n_batch > len — the
        # last batch is short, repeated indices top it up. Per-batch lists
        # here never exceed batch_size (we keep the pad INSIDE the last
        # batch like the official index stream).
        pad = len(self.dataset) - sum(len(b) for b in batches)
        if pad > 0:
            flat = [i for b in batches for i in b]
            flat += (flat * ((pad + len(flat) - 1) // len(flat)))[:pad]
            batches = [flat[i:i + self.batch_size]
                       for i in range(0, len(flat), self.batch_size)]
        return batches

    def __iter__(self):
        for batch in self.batch_indices():
            yield from batch

    def __len__(self) -> int:
        return len(self.dataset)


def batch_sequences(sequences: list[np.ndarray], axis: int = 0,
                    pad_value: int = 0) -> np.ndarray:
    """Official batch_sequences (np.pad + stack), kept for compatibility."""
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    seq_lengths = [s.shape[axis] for s in sequences]
    max_length = max(seq_lengths)
    padded = []
    for s, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + \
                  [(0, 0)] * (ndim - axis - 1)
        padded.append(np.pad(s, padding, mode="constant",
                             constant_values=pad_value))
    return np.stack(padded)
