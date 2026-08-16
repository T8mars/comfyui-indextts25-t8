"""Deterministic pronunciation controls shared by the T8star-Aix integrations.

IndexTTS 2.5 already understands ``<surface|reading>`` annotations.  This
module keeps product-level dictionary handling outside the model object, so a
workflow or desktop request never mutates global inference state.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ANNOTATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
ANNOTATION_CANDIDATE_PATTERN = re.compile(r"<[^>\n]*\|[^>\n]*(?:>|$)")

_PINYIN_BASE_TEXT = """
A AI AN ANG AO
BA BAI BAN BANG BAO BEI BEN BENG BI BIAN BIAO BIE BIN BING BO BU
CA CAI CAN CANG CAO CE CEN CENG CHA CHAI CHAN CHANG CHAO CHE CHEN CHENG CHI CHONG CHOU CHU CHUA CHUAI CHUAN CHUANG CHUI CHUN CHUO CI CONG COU CU CUAN CUI CUN CUO
DA DAI DAN DANG DAO DE DEI DENG DI DIA DIAN DIAO DIE DING DIU DONG DOU DU DUAN DUI DUN DUO
E EI EN ENG ER
FA FAN FANG FEI FEN FENG FO FOU FU
GA GAI GAN GANG GAO GE GEI GEN GENG GONG GOU GU GUA GUAI GUAN GUANG GUI GUN GUO
HA HAI HAN HANG HAO HE HEI HEN HENG HONG HOU HU HUA HUAI HUAN HUANG HUI HUN HUO
JI JIA JIAN JIANG JIAO JIE JIN JING JIONG JIU JU JV JUAN JVAN JUE JVE JUN JVN
KA KAI KAN KANG KAO KE KEN KENG KONG KOU KU KUA KUAI KUAN KUANG KUI KUN KUO
LA LAI LAN LANG LAO LE LEI LENG LI LIA LIAN LIANG LIAO LIE LIN LING LIU LO LONG LOU LU LUAN LUN LUO LV LVE
MA MAI MAN MANG MAO ME MEI MEN MENG MI MIAN MIAO MIE MIN MING MIU MO MOU MU
NA NAI NAN NANG NAO NE NEI NEN NENG NI NIAN NIANG NIAO NIE NIN NING NIU NONG NOU NU NUAN NUO NV NVE
O OU
PA PAI PAN PANG PAO PEI PEN PENG PI PIAN PIAO PIE PIN PING PO POU PU
QI QIA QIAN QIANG QIAO QIE QIN QING QIONG QIU QU QV QUAN QVAN QUE QVE QUN QVN
RAN RANG RAO RE REN RENG RI RONG ROU RU RUA RUAN RUI RUN RUO
SA SAI SAN SANG SAO SE SEN SENG SHA SHAI SHAN SHANG SHAO SHE SHEI SHEN SHENG SHI SHOU SHU SHUA SHUAI SHUAN SHUANG SHUI SHUN SHUO SI SONG SOU SU SUAN SUI SUN SUO
TA TAI TAN TANG TAO TE TENG TI TIAN TIAO TIE TING TONG TOU TU TUAN TUI TUN TUO
WA WAI WAN WANG WEI WEN WENG WO WU
XI XIA XIAN XIANG XIAO XIE XIN XING XIONG XIU XU XV XUAN XVAN XUE XVE XUN XVN
YA YAN YANG YAO YE YI YIN YING YO YONG YOU YU YV YUAN YVAN YUE YVE YUN YVN
ZA ZAI ZAN ZANG ZAO ZE ZEI ZEN ZENG ZHA ZHAI ZHAN ZHANG ZHAO ZHE ZHEI ZHEN ZHENG ZHI ZHONG ZHOU ZHU ZHUA ZHUAI ZHUAN ZHUANG ZHUI ZHUN ZHUO ZI ZONG ZOU ZU ZUAN ZUI ZUN ZUO
"""
PINYIN_BASES = frozenset(_PINYIN_BASE_TEXT.split())
PINYIN_TOKEN_PATTERN = re.compile(r"^([A-ZÜV]+)([1-5])$")
CMU_VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW AX AXR IX UX".split())
CMU_CONSONANTS = frozenset(
    "B CH D DH DX EL EM EN F G HH JH K L M N NG NX P Q R S SH T TH V W WH Y Z ZH".split()
)
KANA_PATTERN = re.compile(r"^[\u3040-\u30ff\u31f0-\u31ffー・\s]+$")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class PronunciationEntry:
    term: str
    reading: str
    language: str = "ZH"
    enabled: bool = True
    case_sensitive: bool = True


@dataclass(frozen=True)
class PronunciationResult:
    text: str
    replacements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class PronunciationValidationError(ValueError):
    pass


def normalize_language(language: str | None, default: str = "ZH") -> str:
    value = str(language or default).strip().upper()
    return value if value in {"ZH", "EN", "JA", "ES", "AR"} else default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "否"}


def normalize_reading(reading: str, language: str) -> str:
    value = " ".join(str(reading or "").strip().split())
    if language == "ZH":
        return value.replace("ü", "V").replace("Ü", "V").upper()
    if language == "EN":
        value = re.sub(r"\s*\.\s*", " . ", value.upper())
        return " ".join(value.split())
    return value


def _load_exact_pinyin_vocab(vocab_path: str | Path | None) -> frozenset[str]:
    if not vocab_path:
        return frozenset()
    path = Path(vocab_path)
    if not path.is_file():
        return frozenset()
    return frozenset(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def validate_reading(reading: str, language: str, *, pinyin_vocab_path: str | Path | None = None):
    language = normalize_language(language)
    normalized = normalize_reading(reading, language)
    warnings: list[str] = []
    errors: list[str] = []
    if not normalized:
        return normalized, (), ("读音不能为空。",)
    if language == "ZH":
        exact_vocab = _load_exact_pinyin_vocab(pinyin_vocab_path)
        for token in normalized.split():
            match = PINYIN_TOKEN_PATTERN.fullmatch(token)
            if not match:
                errors.append(f"中文拼音 {token!r} 必须使用字母加 1–5 声调数字，例如 XING2。")
                continue
            base = match.group(1).replace("Ü", "V")
            if base not in PINYIN_BASES:
                errors.append(f"{token!r} 不是可识别的普通话拼音音节。")
            elif exact_vocab and token not in exact_vocab:
                warnings.append(f"{token!r} 不在当前官方精确词表中，将按合法拼音继续尝试。")
    elif language == "EN":
        tokens = [token for token in normalized.split() if token != "."]
        if not tokens:
            errors.append("英文 CMU 音素不能为空。")
        for token in tokens:
            match = re.fullmatch(r"([A-Z]+)([0-2]?)", token)
            if not match:
                errors.append(f"英文音素 {token!r} 格式无效。")
                continue
            phone, stress = match.groups()
            if phone not in CMU_VOWELS and phone not in CMU_CONSONANTS:
                errors.append(f"{phone!r} 不是支持的 CMU 音素。")
            elif stress and phone not in CMU_VOWELS:
                errors.append(f"辅音 {phone!r} 不能带重音数字。")
    elif language == "JA":
        if not KANA_PATTERN.fullmatch(normalized):
            errors.append("日语读音应使用平假名或片假名。")
    else:
        warnings.append(f"{language} 暂无专用音素校验，将原样传给 IndexTTS 2.5。")
    return normalized, tuple(dict.fromkeys(warnings)), tuple(dict.fromkeys(errors))


def annotation_language(surface: str, default_language: str) -> str:
    default_language = normalize_language(default_language)
    if default_language == "JA" and CJK_PATTERN.search(surface):
        return "JA"
    if CJK_PATTERN.search(surface):
        return "ZH"
    if LATIN_PATTERN.search(surface):
        return "EN"
    return default_language


def entry_from_mapping(item: dict[str, Any], default_language: str = "ZH") -> PronunciationEntry:
    language = normalize_language(item.get("language"), default_language)
    return PronunciationEntry(
        term=str(item.get("term", "")).strip(),
        reading=str(item.get("reading", "")).strip(),
        language=language,
        enabled=_as_bool(item.get("enabled"), True),
        case_sensitive=_as_bool(item.get("case_sensitive"), language != "EN"),
    )


def entries_from_rows(rows: Any, default_language: str = "ZH") -> list[PronunciationEntry]:
    if rows is None:
        return []
    if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        rows = rows.values.tolist()
    result: list[PronunciationEntry] = []
    for row in rows:
        if isinstance(row, dict):
            entry = entry_from_mapping(row, default_language)
        else:
            values = list(row) if isinstance(row, (list, tuple)) else [row]
            values += [None] * (5 - len(values))
            language = normalize_language(values[1], default_language)
            entry = PronunciationEntry(
                term=str(values[0] or "").strip(),
                language=language,
                reading=str(values[2] or "").strip(),
                enabled=_as_bool(values[3], True),
                case_sensitive=_as_bool(values[4], language != "EN"),
            )
        if entry.term or entry.reading:
            result.append(entry)
    return result


def entries_to_rows(entries: Iterable[PronunciationEntry]) -> list[list[Any]]:
    return [[item.term, item.language, item.reading, item.enabled, item.case_sensitive] for item in entries]


def _legacy_entries(data: dict[str, Any], default_language: str) -> list[PronunciationEntry]:
    entries: list[PronunciationEntry] = []
    for term, value in data.items():
        if term in {"version", "entries"}:
            continue
        if isinstance(value, dict):
            for language in ("zh", "en", "ja"):
                reading = value.get(language)
                if reading:
                    entries.append(PronunciationEntry(str(term), str(reading), language.upper(), True, language != "en"))
        else:
            entries.append(PronunciationEntry(str(term), str(value), normalize_language(default_language), True, True))
    return entries


def entries_from_data(data: Any, default_language: str = "ZH") -> list[PronunciationEntry]:
    if data is None:
        return []
    if isinstance(data, dict) and "entries" in data:
        data = data.get("entries") or []
    elif isinstance(data, dict):
        return _legacy_entries(data, default_language)
    if not isinstance(data, list):
        raise PronunciationValidationError("发音词典必须是 entries 列表或术语映射。")
    return [entry_from_mapping(item, default_language) for item in data if isinstance(item, dict)]


def parse_dictionary_text(text: str, default_language: str = "ZH") -> list[PronunciationEntry]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        return entries_from_data(json.loads(raw), default_language)
    except json.JSONDecodeError:
        pass
    try:
        import yaml

        parsed = yaml.safe_load(raw)
        if isinstance(parsed, (dict, list)):
            return entries_from_data(parsed, default_language)
    except Exception:
        pass
    result: list[PronunciationEntry] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) not in (2, 3):
            raise PronunciationValidationError(f"词典第 {line_number} 行格式无效，应为：文字|读音|语言。")
        term, reading = parts[:2]
        language = normalize_language(parts[2] if len(parts) == 3 else default_language, default_language)
        result.append(PronunciationEntry(term, reading, language, True, language != "EN"))
    return result


def dictionary_data(entries: Sequence[PronunciationEntry]) -> dict[str, Any]:
    return {"version": 1, "entries": [asdict(entry) for entry in entries]}


def load_dictionary(path: str | Path, default_language: str = "ZH") -> list[PronunciationEntry]:
    file_path = Path(path)
    return parse_dictionary_text(file_path.read_text(encoding="utf-8"), default_language) if file_path.is_file() else []


def save_dictionary(path: str | Path, entries: Sequence[PronunciationEntry]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = dictionary_data(entries)
    try:
        import yaml

        content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except ImportError:
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(file_path)
    return file_path


def _entry_matches(text: str, position: int, entry: PronunciationEntry) -> bool:
    candidate = text[position : position + len(entry.term)]
    return candidate == entry.term if entry.case_sensitive else candidate.casefold() == entry.term.casefold()


def _apply_entries_to_plain_text(text, entries, *, pinyin_vocab_path):
    replacements: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    output: list[str] = []
    position = 0
    while position < len(text):
        selected = next((entry for entry in entries if _entry_matches(text, position, entry)), None)
        if selected is None:
            output.append(text[position])
            position += 1
            continue
        normalized, item_warnings, item_errors = validate_reading(
            selected.reading, selected.language, pinyin_vocab_path=pinyin_vocab_path
        )
        warnings.extend(f"{selected.term}：{message}" for message in item_warnings)
        if item_errors:
            errors.extend(f"{selected.term}：{message}" for message in item_errors)
            output.append(text[position : position + len(selected.term)])
        else:
            output.append(f"<{text[position:position + len(selected.term)]}|{normalized}>")
            replacements.append(f"{selected.term} → {normalized} ({selected.language})")
        position += len(selected.term)
    return "".join(output), replacements, warnings, errors


def process_pronunciation_text(text, language, entries=None, *, strict=False, pinyin_vocab_path=None):
    source = str(text or "")
    default_language = normalize_language(language)
    warnings: list[str] = []
    errors: list[str] = []
    candidate_spans = list(ANNOTATION_CANDIDATE_PATTERN.finditer(source))
    valid_spans = list(ANNOTATION_PATTERN.finditer(source))
    valid_ranges = {(match.start(), match.end()) for match in valid_spans}
    for candidate in candidate_spans:
        if (candidate.start(), candidate.end()) not in valid_ranges:
            errors.append(f"发音标注格式无效：{candidate.group(0)!r}")
    for match in valid_spans:
        surface, reading = match.groups()
        _normalized, item_warnings, item_errors = validate_reading(
            reading, annotation_language(surface, default_language), pinyin_vocab_path=pinyin_vocab_path
        )
        warnings.extend(f"{surface}：{message}" for message in item_warnings)
        errors.extend(f"{surface}：{message}" for message in item_errors)
    prepared: list[PronunciationEntry] = []
    seen_terms: set[tuple[str, bool]] = set()
    for entry in sorted(entries or (), key=lambda item: len(item.term), reverse=True):
        if not entry.enabled:
            continue
        if not entry.term:
            errors.append("发音词典包含空文字项。")
            continue
        key = (entry.term if entry.case_sensitive else entry.term.casefold(), entry.case_sensitive)
        if key in seen_terms:
            warnings.append(f"词典术语 {entry.term!r} 重复，已使用靠前的一项。")
            continue
        seen_terms.add(key)
        prepared.append(entry)
    replacements: list[str] = []
    output: list[str] = []
    cursor = 0
    for match in valid_spans:
        plain, applied, item_warnings, item_errors = _apply_entries_to_plain_text(
            source[cursor : match.start()], prepared, pinyin_vocab_path=pinyin_vocab_path
        )
        output.extend((plain, match.group(0)))
        replacements.extend(applied)
        warnings.extend(item_warnings)
        errors.extend(item_errors)
        cursor = match.end()
    plain, applied, item_warnings, item_errors = _apply_entries_to_plain_text(
        source[cursor:], prepared, pinyin_vocab_path=pinyin_vocab_path
    )
    output.append(plain)
    replacements.extend(applied)
    warnings.extend(item_warnings)
    errors.extend(item_errors)
    result = PronunciationResult(
        "".join(output), tuple(replacements), tuple(dict.fromkeys(warnings)), tuple(dict.fromkeys(errors))
    )
    if strict and result.errors:
        raise PronunciationValidationError("；".join(result.errors))
    return result


def format_pronunciation_report(result: PronunciationResult) -> str:
    lines = [f"已应用 {len(result.replacements)} 处词典发音。"]
    if result.replacements:
        lines.extend(f"- {item}" for item in result.replacements)
    if result.warnings:
        lines.append("警告：")
        lines.extend(f"- {item}" for item in result.warnings)
    if result.errors:
        lines.append("错误：")
        lines.extend(f"- {item}" for item in result.errors)
    if not result.replacements and not result.warnings and not result.errors:
        lines.append("未发现需要替换或修正的内容；手工发音标注会原样保留。")
    return "\n".join(lines)
