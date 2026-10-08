"""Multilingual bench corpus: per-language + mixed, >100 chars, sentence-split.

Languages covered: zh / en / ja / ko / yue + zh-en / 4-lang mix (zh-en-ja-ko).
Every target text is >100 characters and is split on 。/！/？/. /！/？(and
newlines) before synthesis, mirroring the official cut5-by-sentence semantics.
Reference pairs stay zh (existing ref_zh_3.5s.wav) plus an en reference for
en/mixed cells when available; refs are per-corpus constants so every run is
reproducible. This file is data-only: harnesses import CORPORA.

Usage (harness):
    from tools.bench_corpus import CORPORA, iter_sentences
"""

from __future__ import annotations

import re

# --- reference pairs (fixed for reproducibility) --------------------------
REF_ZH = {
    "audio": "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local/ref_zh_3.5s.wav",
    "text": "希望你以后能够做得比我还好哟。",
    "lang": "zh",
}
# English reference: USES THE SAME zh clip (user decision 2026-10-07) — cross-lingual
# voice clone, prompt_lang=zh with text_lang=en per segment. If a real en
# reference is added later, swap REF_EN's audio/text and lang fields only.
REF_EN = {
    "audio": "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local/ref_zh_3.5s.wav",
    "text": "希望你以后能够做得比我还好哟。",
    "lang": "zh",
}

# --- target corpora (>100 chars each, sentence-split) ----------------------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+|\n+")

CORPORA = {
    # Chinese: 137 chars
    "zh": {
        "text": (
            "大家好，欢迎来到今天的机器学习专题节目。我是你们的主持人，"
            "今天我们要聊一个非常实用的话题，那就是如何在苹果芯片上运行语音合成系统。"
            "首先我们会介绍整个系统的架构设计，然后深入讨论性能优化的思路与方法，"
            "最后再演示几个实际效果。希望这期内容对正在做本地部署的朋友们有所帮助，"
            "也欢迎大家留言告诉我们你想了解的方向。谢谢大家的支持，我们正式开始吧！"
        ),
        "text_lang": "zh",
        "ref": REF_ZH,
    },
    # English: >100 chars
    "en": {
        "text": (
            "Hello everyone, and welcome to today's deep dive into machine learning. "
            "My name is Alex, and I will be your host for this session. "
            "Over the next few minutes, we will explore how modern text to speech systems work, "
            "what makes them fast enough to run on consumer hardware, and why Apple Silicon "
            "is such an interesting platform for local inference. "
            "We will cover model architecture, quantization, and real time performance. "
            "Feel free to take notes, and let's get started right away!"
        ),
        "text_lang": "en",
        "ref": REF_EN,
    },
    # Japanese: >100 chars
    "ja": {
        "text": (
            "皆さん、こんにちは。今日は機械学習についてお話しいたします。"
            "私は司会のアレックスと申します。これから数分間で、最新の音声合成システムが"
            "どのように動いているのか、そしてなぜアップルのチップ上で高速に動作するのかを"
            "解説していきます。モデルの構造から量子化の手法、さらにリアルタイム性能まで、"
            "幅広くカバーする予定です。ぜひ最後までお聞きください。それでは始めましょう。"
        ),
        "text_lang": "ja",
        "ref": REF_ZH,  # ja synthesis from zh prompt (cross-lingual voice clone)
    },
    # Korean: >100 chars
    "ko": {
        "text": (
            "안녕하세요, 여러분. 오늘은 머신러닝에 대해 이야기해 보겠습니다. "
            "저는 진행을 맡은 알렉스입니다. 앞으로 몇 분 동안 최신 음성 합성 시스템이 "
            "어떻게 동작하는지, 그리고 왜 애플 칩에서 빠르게 실행되는지 설명드리겠습니다. "
            "모델 구조부터 양자화 기법, 실시간 성능까지 폭넓게 다룰 예정입니다. "
            "끝까지 들어 주시면 감사하겠습니다. 그럼 시작해 보겠습니다."
        ),
        "text_lang": "ko",
        "ref": REF_ZH,  # ko needs v2 symbol table; cross-lingual clone
    },
    # Cantonese: >100 chars
    "yue": {
        "text": (
            "各位好，歡迎嚟到今日嘅機械學習專題環節。我係你哋嘅主持阿明，"
            "今日同大家傾下點樣喺蘋果芯片上面行語音合成系統。首先我會介紹成個系統嘅架構設計，"
            "跟住再深入討論性能優化嘅思路同方法，最後示範幾個實際效果。"
            "希望呢期內容對做本地部署嘅朋友有幫助，歡迎大家留言話俾我知想了解啲乜。"
            "多謝大家支持，我哋而家正式開始啦！"
        ),
        "text_lang": "yue",
        "ref": REF_ZH,
    },
    # zh-en mixed: >100 chars total
    "zh-en": {
        "text": (
            "大家好，欢迎来到今天的国际开发者大会。Today we are going to talk about "
            "on-device speech synthesis and machine learning. 首先，我们会介绍系统的整体架构，"
            "including the text front end, the autoregressive semantic model, and the vocoder. "
            "然后我们会讨论量化与性能优化，which is the key to running everything locally. "
            "最后还有实际演示环节。Thank you for joining us today, and let's get started!"
        ),
        "text_lang": "auto",  # official mixed mode: per-segment language detect
        "ref": REF_ZH,
    },
    # 4-language mix zh-en-ja-ko: >100 chars total
    "quad": {
        "text": (
            "各位观众大家好，欢迎收看本期节目。Hello everyone, and welcome to the show. "
            "今日は音声合成の最新技術についてお話しします。안녕하세요, 오늘은 음성 합성 기술을 "
            "살펴보겠습니다. 在接下来的时间里，我们会覆盖四种语言的处理流程。We will cover the "
            "full pipeline in four languages. それでは、さっそく始めましょうましょう。"
            "그럼 지금부터 시작하겠습니다. 谢谢大家，我们开始吧！Let's begin!"
        ),
        "text_lang": "auto",
        "ref": REF_ZH,
    },
}


def iter_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation (。！？.!? + newlines), strip empties.

    Mirrors the official by-sentence split semantics (cut5) for long texts:
    each sentence becomes one synthesis segment. Leading/trailing whitespace
    is stripped; pure-punctuation fragments are dropped.
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p and not re.fullmatch(r"[\s!-/:-@\[-`{-~。！？，、；：“”‘’（）《》…—]+", p)] or [text]


def corpus_stats() -> None:
    """Print char/sentence stats per corpus (sanity check, no GPU)."""
    for name, c in CORPORA.items():
        sents = iter_sentences(c["text"])
        n = len(c["text"])
        print(f"{name:8s} chars={n:4d} sentences={len(sents):2d} "
              f"lang={c['text_lang']} ref={c['ref']['lang']}")


if __name__ == "__main__":
    corpus_stats()
