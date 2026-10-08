"""s2 v3/v4/v5 training data pipeline (module/data_utils.py V3/V4 in MLX).

Official semantics ported:

* TextAudioSpeakerLoaderV3 / V4 (module/data_utils.py L279/L517):
  reads ``<exp_dir>/2-name2text.txt`` (name\tphones\tword2ph\tnorm_text),
  intersects names with ``4-cnhubert/<name>.pt|.npy`` and ``5-wav32k/<name>``,
  repeats the list when < 100 items (max(2, 100//n) times), shuffles with
  ``random.seed(1234)``, converts phones via cleaned_text_to_sequence with
  the GLOBAL ``version`` env (official module/data_utils.py L13; the webui
  exports it before spawning s2_train_v3_lora), filters 0.6s < dur < 54s by
  ``os.path.getsize(wav32k) / 32000 / 2``.
* V3 audio: spec32k = spectrogram_torch(2048/640/2048, center=False) on the
  wav32k samples; mel = 100-mel(24k ffmpeg-resampled audio, 1024/256/1024,
  center=False) normalized (x+12)/14*2-1. ssl padded by ONE replicate frame
  when ssl.shape[-1] != spec.shape[-1].
* V4 audio: spec32k same; mel = 100-mel(32k audio, 1280/320/1280,
  center=False), same norm. (v5 family trains with the V4 loader —
  TextAudioSpeakerLoader selection is `V3 if version == "v3" else V4`.)
* CollateV3: sort batch by spec len descending; max_ssl_len1 = 8*((max+8)//8)
  i.e. 8*((max//8)+1); max_ssl_len = 2*((max//2)+1); max_spec_len =
  2*((max//2)+1); max_mel_len = int(max_ssl_len1 * 1.25 * 1.5); mel row width
  max_mel_len.
* CollateV4: max_ssl_len = 2*((max//2)+1); max_spec_len = 2*((max//2)+1);
  mel row width max_spec_len*2.

The 24k ffmpeg resample replicates tools/my_utils.load_audio: ``ffmpeg -i in
-f f32le -acodec pcm_f32le -ac 1 -ar 24000 -`` (bit-parity matters for the
mel target; the loader caches nothing and streams ffmpeg stdout).

Phone id sequences come from the vendored official cleaner
(gsovits_mlx.text.vendored_cpufront) — the same cleaned_text_to_sequence the
official data_utils imports, evaluated in OUR process against the CPUFast
checkout (read-only).
"""

from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass

import numpy as np

from ..text.mel_frontend import mel_spectrogram, spectrogram

# librosa mel (slaney) norm + (x - spec_min) / (spec_max - spec_min) * 2 - 1
SPEC_MIN, SPEC_MAX = -12.0, 2.0


def norm_spec(x):
    return (x - SPEC_MIN) / (SPEC_MAX - SPEC_MIN) * 2 - 1


def ffmpeg_load_audio(path: str, sr: int) -> np.ndarray:
    """tools/my_utils.load_audio: ffmpeg subprocess f32le pcm mono @ sr."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-f", "f32le", "-acodec",
         "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _cleaned_text_to_sequence():
    """Import the vendored official cleaned_text_to_sequence lazily."""
    from ..text import vendored_cpufront as vcf
    return vcf.cleaned_text_to_sequence()


@dataclass
class S2Batch:
    ssl: np.ndarray            # (B, 768, T_ssl) fp32
    spec: np.ndarray           # (B, 1025, T_spec) fp32
    mel: np.ndarray            # (B, 100, T_mel) fp32 (normalized)
    ssl_lengths: np.ndarray    # (B,) int64
    spec_lengths: np.ndarray   # (B,) int64
    text: np.ndarray           # (B, T_text) int64
    text_lengths: np.ndarray   # (B,) int64
    mel_lengths: np.ndarray    # (B,) int64


class TextAudioSpeakerLoaderV3V4:
    """V3/V4/V5 dataset (one class; the only differences are mel params)."""

    def __init__(self, exp_dir: str, version: str, val: bool = False):
        self.version = version
        self.exp_dir = exp_dir
        self.path2 = os.path.join(exp_dir, "2-name2text.txt")
        self.path4 = os.path.join(exp_dir, "4-cnhubert")
        self.path5 = os.path.join(exp_dir, "5-wav32k")
        assert os.path.exists(self.path2), self.path2
        assert os.path.exists(self.path4), self.path4
        assert os.path.exists(self.path5), self.path5

        names4 = set()
        for name in os.listdir(self.path4):
            stem, ext = os.path.splitext(name)
            if ext in (".pt", ".npy"):
                names4.add(stem)
        names5 = set(os.listdir(self.path5))

        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")
        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1]]

        audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
        tmp = audiopaths_sid_text
        leng = len(tmp)
        if leng < 100:
            repeated = []
            for _ in range(max(2, int(100 / leng))):
                repeated += tmp
            audiopaths_sid_text = repeated

        random.seed(1234)
        random.shuffle(audiopaths_sid_text)

        if self.version == "v3":
            # 24k mel front-end (mel comes from an ffmpeg 24k resample)
            self.filter_length_mel = self.win_length_mel = 1024
            self.hop_length_mel = 256
            self.sampling_rate_mel = 24000
        else:  # v4/v5dev/v5turbo share the V4 loader
            self.filter_length_mel = self.win_length_mel = 1280
            self.hop_length_mel = 320
            self.sampling_rate_mel = 32000
        self.n_mel_channels = 100
        self.mel_fmin = 0.0
        self.mel_fmax = None
        self.val = val

        c2s = _cleaned_text_to_sequence()
        new_items = []
        lengths = []
        skipped_phone = skipped_dur = 0
        for audiopath in audiopaths_sid_text:
            try:
                phoneme = self.phoneme_data[audiopath][0].split(" ")
                phoneme_ids = c2s(phoneme, version)
            except Exception:
                skipped_phone += 1
                continue
            size = os.path.getsize(os.path.join(self.path5, audiopath))
            duration = size / 32000.0 / 2
            if duration == 0:
                skipped_dur += 1
                continue
            if 54 > duration > 0.6 or self.val:
                new_items.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * 640))
            else:
                skipped_dur += 1
        print(f"phoneme_data_len: {len(self.phoneme_data)}  "
              f"wav_data_len: {len(new_items)}  "
              f"skipped_phone: {skipped_phone}  skipped_dur: {skipped_dur}")
        assert len(new_items) > 1
        self.audiopaths_sid_text = new_items
        self.lengths = lengths

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def _load_ssl(self, audiopath: str) -> np.ndarray:
        """4-cnhubert/<name>.npy (fp16 (1,768,T)) or official .pt fallback."""
        npy = os.path.join(self.path4, audiopath + ".npy")
        if os.path.exists(npy):
            ssl = np.load(npy)
        else:
            import torch
            ssl = torch.load(os.path.join(self.path4, audiopath + ".pt"),
                             map_location="cpu").numpy()
        return ssl

    def get_audio(self, filename: str):
        audio32 = ffmpeg_load_audio(filename, 32000)
        spec = spectrogram(audio32[None], 2048, 640, 2048, center=False)[0]
        if self.version == "v3":
            audio_mel = ffmpeg_load_audio(filename, 24000)
        else:
            audio_mel = audio32
        mel = mel_spectrogram(
            audio_mel[None], self.filter_length_mel, self.n_mel_channels,
            self.sampling_rate_mel, self.hop_length_mel, self.win_length_mel,
            fmin=self.mel_fmin, fmax=self.mel_fmax, center=False)[0]
        mel = norm_spec(mel)
        return spec, mel

    def __getitem__(self, index: int):
        audiopath, phoneme_ids = self.audiopaths_sid_text[index]
        text = np.asarray(phoneme_ids, dtype=np.int64)
        try:
            spec, mel = self.get_audio(os.path.join(self.path5, audiopath))
            ssl = self._load_ssl(audiopath)
            if ssl.shape[-1] != spec.shape[-1]:
                # official F.pad(ssl.float(), (0,1), mode="replicate")
                pad = np.concatenate([ssl, ssl[..., -1:]], axis=-1)
                ssl = pad.astype(ssl.dtype)
        except Exception:
            import traceback
            traceback.print_exc()
            print("load audio or ssl error!!!!!!", audiopath)
            mel = np.zeros((100, 180 if self.version == "v3" else 192),
                           dtype=np.float32)
            spec = np.zeros((1025, 96), dtype=np.float32)
            ssl = np.zeros((1, 768, 96), dtype=np.float16)
            text = text[-1:]
        return ssl, spec, mel, text


def _collate_common(batch, max_mel_len: int, max_ssl_len: int | None = None,
                    max_spec_len: int | None = None):
    """Shared collate body: sort by spec length descending, zero-pad.

    Padding widths follow the official collates: ssl/spec pads are the
    rounded-up maxima passed in (or raw maxima when omitted); mel pads to
    max_mel_len.
    """
    ids_sorted = sorted(range(len(batch)),
                        key=lambda i: batch[i][1].shape[1], reverse=True)
    if max_ssl_len is None:
        max_ssl_len = max(b[0].shape[2] for b in batch)
    if max_spec_len is None:
        max_spec_len = max(b[1].shape[1] for b in batch)
    max_text_len = max(b[3].shape[0] for b in batch)

    bsz = len(batch)
    ssl_padded = np.zeros((bsz, batch[0][0].shape[1], max_ssl_len),
                          dtype=np.float32)
    spec_padded = np.zeros((bsz, batch[0][1].shape[0], max_spec_len),
                           dtype=np.float32)
    mel_padded = np.zeros((bsz, batch[0][2].shape[0], max_mel_len),
                          dtype=np.float32)
    text_padded = np.zeros((bsz, max_text_len), dtype=np.int64)
    ssl_lengths = np.zeros((bsz,), dtype=np.int64)
    spec_lengths = np.zeros((bsz,), dtype=np.int64)
    text_lengths = np.zeros((bsz,), dtype=np.int64)
    mel_lengths = np.zeros((bsz,), dtype=np.int64)

    for i, idx in enumerate(ids_sorted):
        ssl, spec, mel, text = batch[idx]
        ssl_padded[i, :, : ssl.shape[2]] = ssl[0]
        ssl_lengths[i] = ssl.shape[2]
        spec_padded[i, :, : spec.shape[1]] = spec
        spec_lengths[i] = spec.shape[1]
        mel_padded[i, :, : mel.shape[1]] = mel
        mel_lengths[i] = mel.shape[1]
        text_padded[i, : text.shape[0]] = text
        text_lengths[i] = text.shape[0]
    return S2Batch(ssl=ssl_padded, spec=spec_padded, mel=mel_padded,
                   ssl_lengths=ssl_lengths, spec_lengths=spec_lengths,
                   text=text_padded, text_lengths=text_lengths,
                   mel_lengths=mel_lengths)


def collate_v3(batch):
    """TextAudioSpeakerCollateV3."""
    max_ssl_len = max(b[0].shape[2] for b in batch)
    max_ssl_len1 = 8 * (max_ssl_len // 8 + 1)
    max_ssl_len = 2 * (max_ssl_len // 2 + 1)
    max_spec_len = 2 * (max(b[1].shape[1] for b in batch) // 2 + 1)
    max_mel_len = int(max_ssl_len1 * 1.25 * 1.5)
    return _collate_common(batch, max_mel_len, max_ssl_len=max_ssl_len,
                           max_spec_len=max_spec_len)


def collate_v4(batch):
    """TextAudioSpeakerCollateV4 (v4/v5dev/v5turbo)."""
    max_ssl_len = 2 * (max(b[0].shape[2] for b in batch) // 2 + 1)
    max_spec_len = 2 * (max(b[1].shape[1] for b in batch) // 2 + 1)
    return _collate_common(batch, max_spec_len * 2, max_ssl_len=max_ssl_len,
                           max_spec_len=max_spec_len)


def collate_for(version: str):
    return collate_v3 if version == "v3" else collate_v4


# ---------------------------------------------------------------------------
# bucketing batch sampler (DistributedBucketSampler, single GPU)
# ---------------------------------------------------------------------------

def bucket_batches(lengths: list[int], batch_size: int,
                   boundaries=(32, 300, 400, 500, 600, 700, 800, 900, 1000),
                   rng: random.Random | None = None,
                   seed: int = 1234) -> list[list[int]]:
    """Bucketed batches over spec-length proxies (official boundaries).

    Official DistributedBucketSampler assigns each sample to the smallest
    boundary >= its bucket-by length (the SSL length here; the official
    passes dataset.lengths = size//(2*hop) which is the SPEC length proxy),
    shuffles buckets+inside, and yields consecutive batch_size chunks.
    """
    rng = rng or random.Random(seed)

    def bucket_of(x):
        for i, b in enumerate(boundaries):
            if x <= b:
                return i
        return len(boundaries)

    buckets: dict[int, list[int]] = {}
    for i, x in enumerate(lengths):
        buckets.setdefault(bucket_of(x), []).append(i)
    order = list(buckets.keys())
    rng.shuffle(order)
    batches = []
    for k in order:
        idxs = buckets[k][:]
        rng.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s: s + batch_size]
            if chunk:
                batches.append(chunk)
    rng.shuffle(batches)
    return batches
