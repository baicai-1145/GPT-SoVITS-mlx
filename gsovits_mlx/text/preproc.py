"""Text preprocessing pipeline (official TextPreprocessor port) on the
vendored official modules + MLX BERT/G2PW.

Faithful re-implementation of get_phones_and_bert_with_phone_units and
helpers (pre_seg_text, cut0..cut5, short-text recursion, all 10 language
modes, non-zh all-zero bert). The replicated behaviour list (entry
whitespace-collapapse, pre_seg_text details, per-segment
clean+sequence+BERT, <6-phone recursion, punctuation collapsing, batch-zh
condition) and the deliberate differences (no phone_units exposure, fp32
numpy bert, prompt text handled by callers WITHOUT '。'-prefixing) live
in docs/PARITY_NOTES.md §text preprocessing.
"""

from __future__ import annotations

import os
import time
import re
import sys
from typing import Callable, NamedTuple

import numpy as np

from gsovits_mlx.text import vendored_cpufront

punctuation = set(["!", "?", "…", ",", ".", "-"])  # official module constant

# One resident model, independent of text/results; replace on export changes.
_BERT_RESOURCES: dict[tuple, tuple] = {}


class Segment(NamedTuple):
    phones: list            # cleaned_text_to_sequence ids
    phone_symbols: list     # symbol strings (pre-sequence)
    word2ph: list | None
    norm_text: str
    bert: np.ndarray        # (1024, len(phones)) float32


class FrontEndResult(NamedTuple):
    phones: list            # concatenated phone ids
    bert: np.ndarray        # (1024, T) float32
    norm_text: str
    segments: list[Segment]


class TextFrontend:
    """Lazily-initialized official front-end. Build once per process.

    device: "gpu" (default, Metal) or "cpu" (mx CPU backend — used by the
    CPU-only parity runs so no Metal commands are ever issued).
    """

    def __init__(self, cpufast_repo: str | None = None,
                 models_root: str = "/Volumes/2T/gpt-sovits-models/mlx",
                 device: str = "gpu", lazy: bool = False):
        self.models_root = models_root
        self.device = device
        if device == "cpu":
            import mlx.core as mx

            mx.set_default_device(mx.cpu)
        else:
            _assert_gpu_allowed(device)
        self._bert = None      # MLX BertModel
        self._tok = None       # tokenizers.Tokenizer
        self._zero_cache: dict[int, np.ndarray] = {}
        self._lazy = lazy
        self._splits = None
        self._first_pattern = None
        self._cpufast_repo = cpufast_repo
        if lazy:
            # Goal-B first-call latency: defer the vendored-module load (the
            # heavy part is jieba/pypinyin dict init + English pickle dicts)
            # until the first get_phones_and_bert call.
            pass
        else:
            self._load_official()

    def _load_official(self, cpufast_repo: str | None = None) -> None:
        if self._splits is not None:
            return
        (
            self._clean_text,
            self._clean_text_units,
            self._cleaned_to_seq,
            self._splits,
            self._get_seg_method,
            self._split_big_text,
            self._lang_texts,
        ) = vendored_cpufront.front_end(
            self._cpufast_repo if cpufast_repo is None else cpufast_repo)
        self._first_pattern = "[" + "".join(re.escape(s) for s in self._splits) + "]"
    # ------------------------------------------------------------------
    # BERT (lazy; zh segments only)
    # ------------------------------------------------------------------
    def _load_bert(self):
        if self._bert is not None:
            return self._bert
        import json

        import mlx.core as mx

        from gsovits_mlx.io import load_mlx_safetensors
        from gsovits_mlx.text.bert import BertModel
        from gsovits_mlx.text.bert_tokenizer import (
            find_tokenizer_json, load_bert_tokenizer)

        bert_dir = os.path.join(self.models_root, "bert")
        cache_key = None
        if os.environ.get("GSOVITS_FRONTEND_MODEL_CACHE", "1") == "1":
            paths = [os.path.join(bert_dir, name)
                     for name in ("bert.safetensors", "config.json", "tokenizer.json")]
            files = tuple((os.path.realpath(path), s.st_ino, s.st_mtime_ns, s.st_size)
                          for path in paths for s in (os.stat(path),))
            cache_key = (files, str(mx.default_device()),
                         os.environ.get("GSOVITS_FRONTEND_F32", "0") == "1")
            resident = _BERT_RESOURCES.get(cache_key)
            if resident is not None:
                self._bert, self._tok = resident
                return self._bert
        else:
            _BERT_RESOURCES.clear()
        arrays = load_mlx_safetensors(os.path.join(bert_dir, "bert.safetensors"))
        # GPU frontend: BERT runs in its native fp16 export (CUDA-project
        # bert_model.half() parity). The historical fp32 upcast existed
        # because (1-am)*-1e9 overflows fp16; the mask is clamped to -65504
        # inside BertModel instead. GSOVITS_FRONTEND_F32=1 = old path.
        if os.environ.get("GSOVITS_FRONTEND_F32", "0") != "1":
            arrays = {k: v for k, v in arrays.items() if "position_ids" not in k}
        else:
            arrays = {k: mx.array(np.asarray(v, np.float32))
                      for k, v in arrays.items() if "position_ids" not in k}
        cfg = json.load(open(os.path.join(bert_dir, "config.json")))
        self._bert = BertModel(arrays, cfg)
        if cache_key is not None:
            # A changed export also invalidates tokenizer.json's path cache.
            load_bert_tokenizer.cache_clear()
        self._tok = load_bert_tokenizer(find_tokenizer_json(self.models_root))
        if cache_key is not None:
            _BERT_RESOURCES.clear()
            _BERT_RESOURCES[cache_key] = (self._bert, self._tok)
        return self._bert

    def _bert_feature_zh(self, norm_text: str, word2ph: list) -> np.ndarray:
        bert = self._load_bert()
        import mlx.core as mx

        from gsovits_mlx.text.bert_tokenizer import encode_text

        enc = encode_text(self._tok, norm_text)
        f = bert.get_bert_feature(
            mx.array(enc["input_ids"], mx.int32),
            mx.array(enc["attention_mask"], mx.float32),
            word2ph,
        )
        out = np.array(f, dtype=np.float32)
        assert out.shape[1] == sum(word2ph), (out.shape, sum(word2ph))
        return out

    def _bert_inf(self, lang: str, phones: list, word2ph: list | None,
                  norm_text: str) -> np.ndarray:
        """Official TextPreprocessor.get_bert_inf."""
        if lang == "zh":
            return self._bert_feature_zh(norm_text, word2ph)
        n = len(phones)
        feat = self._zero_cache.get(n)
        if feat is None:
            feat = np.zeros((1024, n), dtype=np.float32)
            if n <= 512:  # official caches with an 8MB budget
                self._zero_cache[n] = feat
        return feat

    # ------------------------------------------------------------------
    # official helpers
    # ------------------------------------------------------------------
    def replace_consecutive_punctuation(self, text: str) -> str:
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        return re.sub(pattern, r"\1", text)

    def _get_first(self, text: str) -> str:
        """Official TextPreprocessor.get_first: the FIRST CLAUSE (text before
        the first split char), stripped — NOT leading split chars. The
        pre_seg_text condition `len(get_first(text)) < 4` then means "the
        first clause is under 4 chars (add a leading 。/.)"; matching only
        leading punctuation would make nearly EVERY sentence prepend and
        diverge from the official phones by one token (found in the
        multilingual bench parity sweep, 2026-10-07)."""
        return re.split(self._first_pattern, text)[0].strip()

    def merge_short_text_in_array(self, texts: list, threshold: int) -> list:
        if len(texts) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def _filter_text(self, texts: list) -> list:
        if all(t in [None, " ", "\n", ""] for t in texts):
            raise ValueError("please enter valid text")
        return [t for t in texts if t not in [None, " ", ""]]

    def pre_seg_text(self, text: str, lang: str,
                     text_split_method: str = "cut0") -> list:
        """Official TextPreprocessor.pre_seg_text (target text ONLY; the
        prompt path must not call this — see segment_prompt)."""
        if self._lazy:
            self._load_official()
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in self._splits and len(self._get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        seg_method = self._get_seg_method(text_split_method)
        text = seg_method(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        _texts = text.split("\n")
        _texts = self._filter_text(_texts)
        _texts = self.merge_short_text_in_array(_texts, 5)
        texts = []
        for t in _texts:
            if len(t.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", t):
                continue  # pure punctuation
            if t[-1] not in self._splits:
                t += "。" if lang != "en" else "."
            if len(t) > 510:
                texts.extend(self._split_big_text(t))
            else:
                texts.append(t)
        return texts

    # ------------------------------------------------------------------
    # per-segment cleaning + bert
    # ------------------------------------------------------------------
    def _clean_segment(self, text: str, lang: str, version: str) -> Segment:
        phones_sym, word2ph, norm_text, _units = self._clean_text_units(
            text, lang, version)
        ids = self._cleaned_to_seq(phones_sym, version)
        bert = self._bert_inf(lang, ids, word2ph, norm_text)
        return Segment(ids, phones_sym, word2ph, norm_text, bert)

    def _can_batch_zh(self, texts: list, lang: str) -> bool:
        if lang not in {"zh", "all_zh"}:
            return False
        if not texts:
            return False
        return all(re.search(r"[A-Za-z]", t) is None for t in texts)

    def _batched_zh_segments(self, texts: list, version: str) -> list[Segment]:
        """Official _preprocess_batch_zh (bert batched; order preserved)."""
        prepared = []
        for t in texts:
            seg = self._clean_zh_short_recursion(t, version)
            if seg is None or seg.norm_text == "":
                continue
            prepared.append(seg)
        # BERT forward per segment: official batches normalized texts into
        # one padded batch; single-sentence forwards produce identical
        # features (attention is masked), so we keep the loop — the batching
        # exists to amortize GPU launches, not to change math.
        return prepared

    def _clean_zh_short_recursion(self, text: str, version: str,
                                  final: bool = False):
        """Official _extract_pure_zh_text: len(phones)<6 -> '.'+text once."""
        seg = self._clean_segment(text, "zh", version)
        if not final and len(seg.phones) < 6:
            return self._clean_zh_short_recursion("." + text, version, True)
        return seg

    # ------------------------------------------------------------------
    # language-mode segmentation (official get_phones_and_bert_with_phone_units)
    # ------------------------------------------------------------------
    def _mode_splits(self, text: str, language: str) -> tuple[list, list]:
        """Returns (textlist, langlist) for the official text_lang modes.

        CPUFast language lists (TTS.py:326-328) — these are the modes that
        exist; upstream's all_zh_mix/all_ja_mix were dropped in CPUFast:
            v1: auto, en, zh, ja, all_zh, all_ja
            v2: auto, auto_yue, en, zh, ja, yue, ko, all_zh, all_ja,
                all_yue, all_ko
        Mixed text under a single-language label (zh/ja/ko/yue) falls into
        the official else branch: non-en runs take the user language
        (因无法区别中日韩文汉字,以用户输入为准), en runs stay en.
        """
        textlist, langlist = [], []
        if language == "all_zh":
            for tmp in self._lang_texts(text, "zh"):
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in self._lang_texts(text, "zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in self._lang_texts(text, "ja"):
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in self._lang_texts(text, "ko"):
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en"); textlist.append(text)
        elif language == "auto":
            for tmp in self._lang_texts(text):
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in self._lang_texts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"]); textlist.append(tmp["text"])
        else:
            for tmp in self._lang_texts(text):
                if langlist:
                    if ((tmp["lang"] == "en" and langlist[-1] == "en")
                            or (tmp["lang"] != "en" and langlist[-1] != "en")):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append("en")
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        return textlist, langlist

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    # official v1/v2 language lists (TTS.py:326-327)
    V1_LANGUAGES = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]
    V2_LANGUAGES = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko",
                    "all_zh", "all_ja", "all_yue", "all_ko"]

    def get_phones_and_bert(self, text: str, language: str, version: str = "v2",
                            final: bool = False) -> FrontEndResult:
        """Official get_phones_and_bert_with_phone_units (minus phone_units)."""
        if self._lazy:
            self._load_official()
        languages = self.V1_LANGUAGES if version == "v1" else self.V2_LANGUAGES
        if language not in languages:
            raise ValueError(
                f"text_lang {language!r} not supported for v{version}; "
                f"choose from {languages}")
        text = re.sub(r" {2,}", " ", text)
        textlist, langlist = self._mode_splits(text, language)

        segments: list[Segment] = []
        for t, lang in zip(textlist, langlist):
            lang = lang.replace("all_", "")
            # unknown languages degrade inside the official cleaner itself
            # (clean_text: language='en', text=' '); do not pre-filter here
            seg = self._clean_segment(t, lang, version)
            if seg.norm_text == "":
                continue
            segments.append(seg)

        phones = [p for s in segments for p in s.phones]
        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version,
                                            final=True)

        bert = np.concatenate([s.bert for s in segments], axis=1) \
            if len(segments) > 1 else segments[0].bert
        norm_text = "".join(s.norm_text for s in segments)
        return FrontEndResult(phones, bert, norm_text, segments)

    def preprocess(self, text: str, lang: str, text_split_method: str = "cut0",
                   version: str = "v2") -> list[FrontEndResult]:
        """Official TextPreprocessor.preprocess (per split-method segment)."""
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        results = []
        if self._can_batch_zh(texts, lang):
            for seg in self._batched_zh_segments(texts, version):
                results.append(FrontEndResult(
                    seg.phones, seg.bert, seg.norm_text, [seg]))
            return results
        for t in texts:
            r = self.get_phones_and_bert(t, lang, version)
            if r.norm_text == "":
                continue
            results.append(r)
        return results

    def segment_prompt(self, ref_text: str, lang: str,
                       version: str = "v2") -> FrontEndResult:
        """Prompt/reference text path — official TTS.py:1316-1324:
        strip('\n'), append '。'/'.' when the text does not end in a split
        separator, then segment_and_extract_feature_for_text
        (get_phones_and_bert, full language-mode semantics; never the
        pre_seg_text '<4-char '。' prefix' rule)."""
        if self._lazy:
            self._load_official()
        text = ref_text.strip("\n")
        if not text:
            raise ValueError("empty prompt text")
        if text[-1] not in self._splits:
            text += "。" if lang != "en" else "."
        return self.get_phones_and_bert(text, lang, version)


def bootstrap(cpufast_repo: str | None = None,
              models_root: str = "/Volumes/2T/gpt-sovits-models/mlx") -> None:
    """chdir into the CPUFast repo (official modules use relative scratch
    paths, e.g. japanese TEMP/ja, g2pw model_dir) before first use, and
    point the official bert_path env (G2PW tokenizer source) at the
    converted bert export, which carries tokenizer.json."""
    repo = vendored_cpufront.cpufast_repo(cpufast_repo)
    if os.getcwd() != repo:
        os.chdir(repo)
    if not os.environ.get("bert_path"):
        os.environ["bert_path"] = os.path.join(models_root, "bert")


def _assert_gpu_allowed(device: str) -> None:
    """GPU-lock discipline (lead mandate, task-2): selecting the Metal device
    requires holding the CANONICAL main-repo .tmp/gpu.lock.d (owner file
    naming this agent/task).

    2026-10-09 postmortem: accepting the first lock found while walking up
    from here let worktree-local locks authorize GPU use, breaking mutual
    exclusion (3 concurrent trainers -> machine OOM -> watchdog reboot).
    Delegating to gsovits_mlx.gpu_lock, whose resolution is cwd-independent
    (anchored at the package location, topmost .git-bearing ancestor = main
    repo root) — the bootstrap chdir into CPUFast no longer matters.
    """
    if device != "gpu":
        return
    from gsovits_mlx.gpu_lock import _find_lock, LOCK_STALE_SEC

    found = _find_lock()
    if found is None:
        raise RuntimeError(
            "GPU device requested but no canonical gpu.lock.d found at the "
            "main-repo .tmp; claim the lock (gpu_lock.acquire_lock) before "
            "Metal runs")
    owner, lock = found
    # Machine law: the lock is ANONYMOUS discipline -- any fresh, non-empty
    # owner counts. Do NOT name-match specific agents.
    try:
        fresh = (time.time()
                 - os.path.getmtime(os.path.join(lock, "owner"))) < LOCK_STALE_SEC
    except OSError:
        fresh = False
    if owner and fresh:
        return
    raise RuntimeError(
        f"GPU device requested but gpu.lock is stale/empty "
        f"(owner={owner!r}, fresh={fresh}); refresh the owner file "
        "or claim the lock")
