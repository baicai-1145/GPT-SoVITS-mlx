"""Language segmentation compatible with GPT-SoVITS-CPUFast `LangSegmenter`.

The official implementation (GPT_SoVITS/text/LangSegmenter/langsegmenter.py)
wraps the `split_lang` package and post-processes its output:

* "digit" runs get the caller's default language (or a context rule when no
  default is given),
* runs the LID model mis-detected but that are pure-ASCII English are forced
  to "en" (full_en),
* non-ja runs are re-split for embedded kana sequences, then non-ko runs for
  embedded hangul (split_jako),
* unknown ("x") runs become zh when they contain CJK Han glyphs (full_cjk),
* leftover "x" runs inherit a neighbour's language (or zh as last resort).

This module re-implements getTexts() faithfully on top of `split_lang`, so
the runtime needs NO fast_langdetect import at all. The official package
cannot be imported without it: LangSegmenter/langsegmenter.py imports
fast_langdetect UNGUARDED at module top (only to redirect its default
detector's model cache directory) and never calls the LID model on the
call chain (all classification decisions come from split_lang). When
fast_langdetect IS importable and a cache_dir is provided, configure()
performs the same _default_detector swap so any other consumer of
fast_langdetect resolves the same model cache as the official checkout.

Verified equivalent to the official LangSegmenter.getTexts on the P2-1
acceptance corpus (see tests/test_lang_segmenter.py).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

__all__ = ["get_texts", "configure", "DEFAULT_LANG_MAP"]

# Official LangSegmenter.DEFAULT_LANG_MAP (gsv four-language filter)
DEFAULT_LANG_MAP = {
    "zh": "zh",
    "yue": "zh",  # 粤语
    "wuu": "zh",  # 吴语
    "zh-cn": "zh",
    "zh-tw": "x",  # 繁体设置为x
    "ko": "ko",
    "ja": "ja",
    "en": "en",
}

_full_en_re = re.compile(
    r"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$"
)
_punct_for_cjk = r"[0-9、-〜。！？.!?… /]+$"

# CJK Unified Ideographs + Extensions A..H (official list from the wiki)
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DB5),
    (0x20000, 0x2A6DD),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
    (0x2EBF0, 0x2EE5D),
)

_JA_RE = re.compile(
    r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+"
    r"(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
)
_KO_RE = re.compile(
    r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+"
    r"(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"
)

_SENT_END = [",", ".", "!", "?", "，", "。", "！", "？"]


def _is_full_en(text: str) -> bool:
    return bool(_full_en_re.match(text))


def _full_cjk(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES) or re.match(_punct_for_cjk, ch):
            out.append(ch)
    return "".join(out)


def _split_jako(tag_lang: str, item: Dict[str, str]) -> List[Dict[str, str]]:
    pattern = _JA_RE if tag_lang == "ja" else _KO_RE
    lang_list: List[Dict[str, str]] = []
    tag = 0
    for match in pattern.finditer(item["text"]):
        if match.start() > tag:
            lang_list.append({"lang": item["lang"], "text": item["text"][tag:match.start()]})
        tag = match.end()
        lang_list.append({"lang": tag_lang, "text": item["text"][match.start():match.end()]})
    if tag < len(item["text"]):
        lang_list.append({"lang": item["lang"], "text": item["text"][tag:len(item["text"])]})
    return lang_list


def _merge_lang(lang_list: List[Dict[str, str]], item: Dict[str, str]) -> List[Dict[str, str]]:
    if lang_list and item["lang"] == lang_list[-1]["lang"]:
        lang_list[-1]["text"] += item["text"]
    else:
        lang_list.append(item)
    return lang_list


def _split_by_lang(text: str) -> List[Dict[str, str]]:
    """split_lang LangSplitter.split_by_lang with the official settings."""
    from split_lang import LangSplitter

    splitter = LangSplitter(lang_map=DEFAULT_LANG_MAP)
    splitter.merge_across_digit = False
    substr = splitter.split_by_lang(text=text)
    return [{"lang": item.lang, "text": item.text} for item in substr]


def get_texts(text: str, default_lang: str = "") -> List[Dict[str, str]]:
    """Official LangSegmenter.getTexts(text, default_lang)."""
    substr = _split_by_lang(text)

    lang_list: List[Dict[str, str]] = []
    have_num = False

    for item in substr:
        dict_item = {"lang": item["lang"], "text": item["text"]}

        if dict_item["lang"] == "digit":
            if default_lang != "":
                dict_item["lang"] = default_lang
            else:
                have_num = True
            lang_list = _merge_lang(lang_list, dict_item)
            continue

        # 处理短英文被识别为其他语言的问题
        if _is_full_en(dict_item["text"]):
            dict_item["lang"] = "en"
            lang_list = _merge_lang(lang_list, dict_item)
            continue

        if default_lang != "":
            dict_item["lang"] = default_lang
            lang_list = _merge_lang(lang_list, dict_item)
            continue
        else:
            # 处理非日语夹日文的问题(不包含CJK)
            ja_list: List[Dict[str, str]] = []
            if dict_item["lang"] != "ja":
                ja_list = _split_jako("ja", dict_item)
            if not ja_list:
                ja_list.append(dict_item)

            # 处理非韩语夹韩语的问题(不包含CJK)
            temp_list: List[Dict[str, str]] = []
            for ko_item in ja_list:
                ko_list: List[Dict[str, str]] = []
                if ko_item["lang"] != "ko":
                    ko_list = _split_jako("ko", ko_item)
                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            # 未存在非日韩文夹日韩文
            if len(temp_list) == 1:
                if dict_item["lang"] == "x":
                    cjk_text = _full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                        lang_list = _merge_lang(lang_list, dict_item)
                    else:
                        lang_list = _merge_lang(lang_list, dict_item)
                    continue
                else:
                    lang_list = _merge_lang(lang_list, dict_item)
                    continue

            # 存在非日韩文夹日韩文
            for temp_item in temp_list:
                if temp_item["lang"] == "x":
                    cjk_text = _full_cjk(temp_item["text"])
                    if cjk_text:
                        lang_list = _merge_lang(lang_list, {"lang": "zh", "text": cjk_text})
                    else:
                        lang_list = _merge_lang(lang_list, temp_item)
                else:
                    lang_list = _merge_lang(lang_list, temp_item)

    # 有数字
    if have_num:
        temp_list = lang_list
        lang_list = []
        for i, temp_item in enumerate(temp_list):
            if temp_item["lang"] == "digit":
                if default_lang:
                    temp_item["lang"] = default_lang
                elif lang_list and i == len(temp_list) - 1:
                    temp_item["lang"] = lang_list[-1]["lang"]
                elif not lang_list and i < len(temp_list) - 1:
                    temp_item["lang"] = temp_list[1]["lang"]
                elif lang_list and i < len(temp_list) - 1:
                    if lang_list[-1]["lang"] == temp_list[i + 1]["lang"]:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif lang_list[-1]["text"][-1] in _SENT_END:
                        temp_item["lang"] = temp_list[i + 1]["lang"]
                    elif temp_list[i + 1]["text"][0] in _SENT_END:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif temp_item["text"][-1] in ["。", "."]:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif len(lang_list[-1]["text"]) >= len(temp_list[i + 1]["text"]):
                        temp_item["lang"] = lang_list[-1]["lang"]
                    else:
                        temp_item["lang"] = temp_list[i + 1]["lang"]
                else:
                    temp_item["lang"] = "zh"
            lang_list = _merge_lang(lang_list, temp_item)

    # 筛X
    temp_list = lang_list
    lang_list = []
    for temp_item in temp_list:
        if temp_item["lang"] == "x":
            if lang_list:
                temp_item["lang"] = lang_list[-1]["lang"]
            elif len(temp_list) > 1:
                temp_item["lang"] = temp_list[1]["lang"]
            else:
                temp_item["lang"] = "zh"
        lang_list = _merge_lang(lang_list, temp_item)

    return lang_list


_configured = False


def configure(cache_dir: Optional[str] = None) -> None:
    """Point fast_langdetect's default detector at `cache_dir` (official
    langsegmenter.py does this unconditionally at import time). No-op when
    fast_langdetect is unavailable or lacks the `infer` submodule layout."""
    global _configured
    if _configured or cache_dir is None:
        return
    try:
        import fast_langdetect
        from pathlib import Path

        fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
            fast_langdetect.infer.LangDetectConfig(cache_dir=Path(cache_dir))
        )
        _configured = True
    except Exception:
        _configured = True  # absent package: nothing to configure, don't retry
