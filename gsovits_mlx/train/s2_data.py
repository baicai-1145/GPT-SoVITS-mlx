"""s2 (SoVITS v1/v2/v2Pro/v2ProPlus) training data pipeline.

Port of GPT-SoVITS module/data_utils.py TextAudioSpeakerLoader /
TextAudioSpeakerCollate / DistributedBucketSampler (single-process; the
official num_replicas=1 CPU case) reading the exp_dir produced by
tools/prepare_data.py (task-2 layout: .npy feature files instead of torch .pt).

GAN-trainer memory additions (this port, OFF by default — see
pad_multiple/fixed_widths below; --pad-multiple / --fixed-shape in
tools/train_s2.py): quantized or fully-fixed collated widths recycle a
small set of Metal buffer sizes instead of fragmenting the allocator
cache with ~1 distinct width per batch (measured 83 distinct spec widths
in 120 b3 batches of the 939-item exp set). Extra pad is zero columns
beyond every *_lengths entry; all loss/mask consumption paths are
length-bounded (y_mask/x_mask from sequence_mask, kl summed under z_mask,
segment slices in-bounds), so losses are pad-neutral EXCEPT the posterior
randn draw, whose width follows the padded shape exactly like official
torch (torch.randn over the padded tensor) — fixed/quantized widths shift
the draw in the SAME way official torch pads do; NOT a semantic deviation.

Semantics replicated exactly (verified against the official source):
  * dataset construction: intersect 2-name2text ∩ 4-cnhubert ∩ 5-wav32k
    (+7-sv_cn for Pro), the <100-file duplication loop, random.seed(1234)
    shuffle of the ORDER (python stdlib — bit-exact across implementations),
    duration filter 0.6 < d < 54 via file size / (32000*2), lengths =
    size // (2*hop).
  * __getitem__: load 32 kHz wav (int16 PCM from the prep pipeline) ->
    float32 /32768; spectrogram via the MLX stft front-end (hann periodic,
    reflect pad (n_fft-hop)/2, center=False — parity-tested vs torch);
    ssl .npy load with F.pad(ssl, (0,1), "replicate") when the ssl frame
    count != spec frame count (official quirk).
  * collate: ids sorted by spec_len DESC (stable), max_ssl_len =
    2*((max//2)+1), max_spec_len likewise, zero-pad everything.
  * bucket sampler: boundaries [32,300,...,1900], per-bucket repeat to a
    multiple of batch_size, in-bucket sequential batches, batch-level shuffle.
    RNG: the official torch.randperm(generator=Philox(seed=epoch)) is NOT
    bit-reproducible outside torch; we use numpy default_rng(epoch) and
    DOCUMENT the divergence (see BUCKET RNG note below).

BUCKET RNG note (deviation from official):
    torch.Generator with manual_seed(epoch) produces a Philox counter stream;
    numpy's PCG64 stream differs. Batch ORDER and the repeated-id tail
    composition therefore differ from a torch run at the same epoch. This is
    unavoidable without reimplementing Philox; the sampler is still
    deterministic per epoch (same epoch -> same batches on this
    implementation), every sample appears in exactly one batch, and the
    dataset-level shuffle (seed 1234, stdlib random) IS bit-exact with the
    official code. Real official runs also reshuffle per-process (list(set())
    ordering is PYTHONHASHSEED-dependent), so batch composition is not a
    parity target in the first place.
"""

from __future__ import annotations

import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextAudioSpeakerLoader:
    """s2 training dataset over an exp_dir (task-2 prep layout)."""

    def __init__(self, exp_dir: str, version: str = "v2",
                 sampling_rate: int = 32000, filter_length: int = 2048,
                 hop_length: int = 640, win_length: int = 2048,
                 symbol_to_id: dict | None = None):
        from gsovits_mlx.text import phone_tables

        self.path2 = os.path.join(exp_dir, "2-name2text.txt")
        self.path4 = os.path.join(exp_dir, "4-cnhubert")
        self.path5 = os.path.join(exp_dir, "5-wav32k")
        assert os.path.exists(self.path2), self.path2
        assert os.path.exists(self.path4), self.path4
        assert os.path.exists(self.path5), self.path5
        self.is_v2pro = version in ("v2Pro", "v2ProPlus")
        if self.is_v2pro:
            self.path7 = os.path.join(exp_dir, "7-sv_cn")
            assert os.path.exists(self.path7), self.path7
        names4 = {name[:-4] for name in os.listdir(self.path4)}  # .npy suffix
        names5 = set(os.listdir(self.path5))
        if self.is_v2pro:
            names6 = {name[:-4] for name in os.listdir(self.path7)}
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        if symbol_to_id is None:
            symbol_to_id = phone_tables.symbol_to_id("v1" if version == "v1" else "v2")
        self.symbol_to_id = symbol_to_id

        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]
        if self.is_v2pro:
            self.audiopaths_sid_text = list(
                set(self.phoneme_data) & names4 & names5 & names6)
        else:
            self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = self.audiopaths_sid_text
        leng = len(tmp)
        min_num = 100
        if leng < min_num and leng > 0:
            self.audiopaths_sid_text = []
            for _ in range(max(2, int(min_num / leng))):
                self.audiopaths_sid_text += tmp

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in self.audiopaths_sid_text:
            phoneme = self.phoneme_data[audiopath][0]
            phoneme = phoneme.split(" ")
            try:
                phoneme_ids = [self.symbol_to_id[s] for s in phoneme if s != ""]
                assert len(phoneme_ids) > 0
            except (KeyError, AssertionError):
                skipped_phone += 1
                continue
            size = os.path.getsize(os.path.join(self.path5, audiopath))
            duration = size / self.sampling_rate / 2
            if duration == 0:
                skipped_dur += 1
                continue
            if 54 > duration > 0.6:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1
                continue
        assert len(audiopaths_sid_text_new) > 1
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.stats = dict(n_phoneme_data=len(self.phoneme_data),
                          n_items=len(audiopaths_sid_text_new),
                          skipped_phone=skipped_phone, skipped_dur=skipped_dur)

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def _spectrogram(self, audio: np.ndarray) -> np.ndarray:
        import mlx.core as mx
        from gsovits_mlx.text.mel_frontend import stft_magnitude
        spec = stft_magnitude(mx.array(audio[None]), self.filter_length,
                              self.hop_length, self.win_length)
        return np.asarray(spec[0], dtype=np.float32)

    def get_audio(self, filename: str):
        import soundfile as sf
        audio, sr = sf.read(filename, dtype="float32")
        assert sr == self.sampling_rate, (sr, filename)
        # prep pipeline wrote int16 PCM; soundfile float32 read == int16/32768
        spec = self._spectrogram(audio)
        return spec, audio

    def __getitem__(self, index: int):
        audiopath, phoneme_ids = self.audiopaths_sid_text[index]
        text = np.asarray(phoneme_ids, np.int64)
        try:
            spec, wav = self.get_audio(os.path.join(self.path5, audiopath))
            ssl = np.load(os.path.join(self.path4, audiopath + ".npy"))
            if ssl.shape[-1] != spec.shape[-1]:
                ssl = np.pad(ssl.astype(np.float32), ((0, 0), (0, 0), (0, 1)),
                             mode="edge")
            if self.is_v2pro:
                sv_emb = np.load(os.path.join(self.path7, audiopath + ".npy"))
        except Exception:
            import traceback
            traceback.print_exc()
            spec = np.zeros((1025, 100), np.float32)
            wav = np.zeros(100 * self.hop_length, np.float32)
            ssl = np.zeros((1, 768, 100), np.float32)
            text = text[-1:]
            if self.is_v2pro:
                sv_emb = np.zeros((1, 20480), np.float32)
            print("load audio or ssl error!!!!!!", audiopath)
        if self.is_v2pro:
            return (ssl, spec, wav, text, sv_emb)
        return (ssl, spec, wav, text)


class TextAudioSpeakerCollate:
    def __init__(self, version: str = "v2", pad_multiple: int | None = None,
                 fixed_widths: dict | None = None):
        self.is_v2pro = version in ("v2Pro", "v2ProPlus")
        self.pad_multiple = pad_multiple
        self.fixed_widths = fixed_widths  # {name: width} overrides ALL maxima

    def __call__(self, batch):
        # ids sorted by spec length desc (torch.sort stable for equal keys)
        order = sorted(range(len(batch)), key=lambda i: -batch[i][1].shape[-1])
        max_ssl_len = max(x[0].shape[2] for x in batch)
        max_ssl_len = int(2 * ((max_ssl_len // 2) + 1))
        max_spec_len = max(x[1].shape[1] for x in batch)
        max_spec_len = int(2 * ((max_spec_len // 2) + 1))
        max_wav_len = max(x[2].shape[0] for x in batch)
        max_text_len = max(x[3].shape[0] for x in batch)
        if self.pad_multiple:
            m = self.pad_multiple
            max_ssl_len = ((max_ssl_len + m - 1) // m) * m
            max_spec_len = ((max_spec_len + m - 1) // m) * m
            # hop 640: wav width stays spec-aligned (20480 % 640 == 0)
            max_wav_len = ((max_wav_len + m * 640 - 1) // (m * 640)) * (m * 640)
            max_text_len = ((max_text_len + m - 1) // m) * m
        if self.fixed_widths:
            fw = self.fixed_widths
            max_ssl_len = fw.get("ssl", max_ssl_len)
            max_spec_len = fw.get("spec", max_spec_len)
            max_wav_len = fw.get("wav", max_wav_len)
            max_text_len = fw.get("text", max_text_len)

        n = len(batch)
        n_freq = batch[0][1].shape[0]
        ssl_ch = batch[0][0].shape[1]
        ssl_padded = np.zeros((n, ssl_ch, max_ssl_len), np.float32)
        spec_padded = np.zeros((n, n_freq, max_spec_len), np.float32)
        wav_padded = np.zeros((n, 1, max_wav_len), np.float32)
        text_padded = np.zeros((n, max_text_len), np.int64)
        ssl_lengths = np.zeros(n, np.int64)
        spec_lengths = np.zeros(n, np.int64)
        wav_lengths = np.zeros(n, np.int64)
        text_lengths = np.zeros(n, np.int64)
        if self.is_v2pro:
            sv_embs = np.zeros((n, 20480), np.float32)

        for i, idx in enumerate(order):
            row = batch[idx]
            ssl, spec, wav, text = row[0], row[1], row[2], row[3]
            ssl_padded[i, :, : ssl.shape[2]] = ssl[0]
            ssl_lengths[i] = ssl.shape[2]
            spec_padded[i, :, : spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav_padded[i, 0, : wav.shape[0]] = wav
            wav_lengths[i] = wav.shape[0]
            text_padded[i, : text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            if self.is_v2pro:
                sv = row[4]
                sv_embs[i] = sv.reshape(-1)[:20480]

        out = (ssl_padded, ssl_lengths, spec_padded, spec_lengths,
               wav_padded, wav_lengths, text_padded, text_lengths)
        if self.is_v2pro:
            out = out + (sv_embs,)
        return out


# ---------------------------------------------------------------------------
# Bucket sampler (single replica)
# ---------------------------------------------------------------------------

DEFAULT_BOUNDARIES = [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
                      1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]


class BucketSampler:
    """DistributedBucketSampler with num_replicas=1.

    Divergence from torch: randperm uses numpy default_rng(epoch) instead of
    torch Philox(generator seeded with epoch) — see module docstring. The
    dataset-level order (seed-1234 stdlib shuffle) IS official-exact.
    """

    def __init__(self, lengths, batch_size: int, boundaries=None, shuffle: bool = True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.boundaries = list(boundaries or DEFAULT_BOUNDARIES)
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size  # num_replicas=1

    def _bisect(self, x):
        boundaries = self.boundaries
        lo, hi = 0, len(boundaries) - 1
        while hi > lo:
            mid = (hi + lo) // 2
            if boundaries[mid] < x <= boundaries[mid + 1]:
                return mid
            elif x <= boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return -1

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i, length in enumerate(self.lengths):
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1
        num_samples_per_bucket = []
        for bucket in buckets:
            len_bucket = len(bucket)
            total_batch_size = self.batch_size  # num_replicas=1
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(rng.permutation(len(bucket)).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
        batches = []
        for i, bucket in enumerate(self.buckets):
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket = (ids_bucket + ids_bucket * (rem // len_bucket)
                          + ids_bucket[: (rem % len_bucket)])
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in
                         ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)
        if self.shuffle:
            batch_ids = rng.permutation(len(batches)).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        return iter(batches)

    def __len__(self):
        return self.num_samples // self.batch_size
