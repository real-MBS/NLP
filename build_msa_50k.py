"""Clean raw SDAIA MSA files and build up to 50,000 training pairs."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import unicodedata

#import the phrases and switches from a separate file for easier maintenance
from dictionaries import PHRASES, SWITCHES


ROOT = Path(__file__).resolve().parent
GOAL = 50_000
WORD = r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)*"
ENGLISH = re.compile(WORD + r"(?:[ -]+" + WORD + r")*")
ARABIC_LETTER = re.compile(r"[ء-ي]")
ARTICLE = re.compile(r"(?:^|[^\w])(ال|لل) ?$")
TIME = re.compile(
    r"(?<![\w.:])(at )?(\d{1,2}(?::\d{2})?) ?([ap])"
    r"(?:m\b|\.m\b(?:\.(?= [^\W\d_]))?)",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:[.,:٫٬/-]\d+)*")


# Arabic diacritics count as part of a word at switch boundaries.
ARABIC_WORD = r"\w\u064b-\u065f\u0670"
OPTIONS = "|".join(re.escape(phrase) for phrase in sorted(SWITCHES, key=len, reverse=True))
PATTERN = re.compile(rf"(?<![{ARABIC_WORD}])(?:{OPTIONS})(?![{ARABIC_WORD}])")
ARABIC = re.compile(r"[ء-ي]")


def clean_text(text):
    # Remove invisible marks that control how mixed text is displayed.
    text = re.sub(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)

    # These are the five decorative symbols found in the MSA files.
    text = re.sub(r"[☀☕✈✦🏖]\ufe0f?", " ", text)

    # Replace consecutive spaces or tabs with one space, then trim the edges.
    return re.sub(r"\s+", " ", text).strip()


def clean_split(source_path, output_path):
    seen = set()
    stats = {
        "read": 0,
        "kept": 0,
        "blank_after_cleaning": 0,
        "duplicate": 0,
        "changed_and_kept": 0,
        "examples": [],
    }
    original_lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    stats["read"] = len(original_lines)
    cleaned_lines = []

    for line_number, original in enumerate(original_lines, start=1):
        cleaned = clean_text(original)
        if not cleaned:
            stats["blank_after_cleaning"] += 1
            continue
        if cleaned in seen:
            stats["duplicate"] += 1
            continue

        seen.add(cleaned)
        cleaned_lines.append(cleaned + "\n")
        if cleaned != original:
            stats["changed_and_kept"] += 1
            if len(stats["examples"]) < 5:
                stats["examples"].append(
                    {"line": line_number, "before": original, "after": cleaned}
                )

    stats["kept"] = len(cleaned_lines)
    output_path.write_text("".join(cleaned_lines), encoding="utf-8")
    return stats, seen


def translate_time(match):
    at, number, period = match.groups()
    prefix = "في الساعة " if at else ""
    suffix = "صباحًا" if period.lower() == "a" else "مساءً"
    return prefix + number + " " + suffix


def make_target(source, phrases):
    if not ARABIC_LETTER.search(source):
        return None, "no_arabic", []

    text, time_count = TIME.subn(translate_time, source)
    matches = list(ENGLISH.finditer(text))
    if not matches and not time_count:
        return None, "no_english", []

    missing = []
    parts = []
    end = 0
    for match in matches:
        key = match[0].casefold().replace("’", "'")
        start = match.start()
        before = text[:start]
        after = text[match.end():]
        article = ARTICLE.search(before)
        arabic = phrases.get(key)
        if article:
            arabic = phrases.get("the " + key)
            if arabic is None or not arabic.startswith("ال"):
                return None, "unresolved_arabic_article", [article[1] + " " + key]
            if article[1] == "لل":
                arabic = "ل" + arabic[1:]
            start = article.start(1)
        # A switch attached to an Arabic word or a number needs its own rule.
        if (not article and before and before[-1].isalnum()) or (after and after[0].isalnum()):
            return None, "attached_switch", []

        if arabic is None:
            missing.append(key)
        else:
            # Avoid obvious duplicated prepositions at a switch boundary.
            if re.search(r"(?:^|\s)(إلى|في|مع) $", before) and arabic.startswith(
                before.split()[-1] + " "
            ):
                return None, "repeated_preposition", []
            if key == "i think" and re.match(r" أن\b", after):
                arabic = "أعتقد"
            parts.extend((text[end:start], arabic))
        end = match.end()

    if missing:
        return None, "missing_phrase", missing

    parts.append(text[end:])
    target = "".join(parts)
    if re.search(r"\bI enjoy\b", source, re.IGNORECASE) or "أستمتع " in source:
        target = re.sub(
            r"\bأستمتع (قراءة|تعلّم|السفر|الاستماع|مشاهدة)\b", r"أستمتع ب\1", target
        )
    if re.search(r"\bthe conference ستكون\b", source, re.IGNORECASE):
        target = target.replace("المؤتمر ستكون", "المؤتمر سيكون", 1)
    if any(char.isalpha() and "ARABIC" not in unicodedata.name(char, "") for char in target):
        return None, "remaining_non_arabic_letters", []
    if NUMBER.findall(source) != NUMBER.findall(target):
        return None, "number_mismatch", []
    return target, "paired", []


def build_dictionary_pairs(train, test, output):
    counts = Counter()
    missing_phrases = Counter()

    with (output / "train_pairs.jsonl").open("w", encoding="utf-8") as pairs:
        for line_number, source in enumerate(train, 1):
            if source in test:
                counts["excluded_test_overlap"] += 1
                continue
            target, status, missing = make_target(source, PHRASES)
            counts[status] += 1
            missing_phrases.update(missing)
            if target is not None:
                if NUMBER.search(source):
                    counts["paired_with_numbers"] += 1
                row = {"source_line": line_number, "source": source, "target_ar": target}
                pairs.write(json.dumps(row, ensure_ascii=False) + "\n")

    eligible = len(train) - counts["excluded_test_overlap"]
    report = {
        "cleaned_train_rows": len(train),
        "eligible_train_rows": eligible,
        "dictionary_entries": len(PHRASES),
        "coverage_percent": round(100 * counts["paired"] / eligible, 2) if eligible else 0,
        "rows_by_status": {key: value for key, value in counts.items() if key != "paired_with_numbers"},
        "number_check": {
            "pairs_checked": counts["paired"],
            "pairs_with_digit_strings": counts["paired_with_numbers"],
            "mismatches_rejected": counts["number_mismatch"],
        },
        "distinct_unresolved_phrases_logged": len(missing_phrases),
        "common_missing_phrases": missing_phrases.most_common(50),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "unresolved_phrases.json").write_text(
        json.dumps(dict(missing_phrases.most_common()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def variants(target):
    matches = [
        match for match in PATTERN.finditer(target)
        if match[0] != "الاجتماع" or not target[:match.start()].endswith("يمكننا ")
    ]
    for size in (1, 2):
        for selected in combinations(matches, size):
            source = target
            switches = []
            for match in reversed(selected):
                english = SWITCHES[match[0]]
                source = source[:match.start()] + english + source[match.end():]
                switches.append({"ar": match[0], "en": english})
            if ARABIC.search(source) and NUMBER.findall(source) == NUMBER.findall(target):
                yield source, list(reversed(switches))


def expand_pairs(train, test, output):
    original = [
        json.loads(line)
        for line in (output / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    seen = set(train) | test
    synthetic = []

    # Prefer one new input per target before taking a second or third.
    by_target = {row["target_ar"]: row for row in reversed(original)}
    candidates = list(by_target.values())
    random.Random(42).shuffle(candidates)
    choices = [(row, iter(variants(row["target_ar"]))) for row in candidates]
    for _ in range(3):
        for row, options in choices:
            for source, switches in options:
                if source in seen:
                    continue
                seen.add(source)
                synthetic.append({
                    "source": source,
                    "target_ar": row["target_ar"],
                    "origin": "synthetic_switch",
                    "parent_source_line": row["source_line"],
                    "switches": switches,
                })
                break
            if len(original) + len(synthetic) >= GOAL:
                break
        if len(original) + len(synthetic) >= GOAL:
            break

    combined = [{**row, "origin": "sdaia_dictionary"} for row in original] + synthetic
    with (output / "train_pairs_50k.jsonl").open("w", encoding="utf-8") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "requested_pairs": GOAL,
        "total_pairs": len(combined),
        "pairs_by_origin": dict(Counter(row["origin"] for row in combined)),
        "distinct_targets": len(by_target),
        "pairs_with_numbers": sum(bool(NUMBER.search(row["source"])) for row in combined),
        "number_mismatches": sum(
            NUMBER.findall(row["source"]) != NUMBER.findall(row["target_ar"])
            for row in combined
        ),
        "exact_test_overlaps": sum(row["source"] in test for row in combined),
    }
    (output / "expansion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report["total_pairs"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", nargs="?", type=Path, default=ROOT / "msa_data")
    parser.add_argument("output_folder", nargs="?", type=Path, default=ROOT / "msa_results")
    args = parser.parse_args()

    if args.output_folder.exists():
        parser.error("Choose a new output folder.")

    try:
        with TemporaryDirectory(dir=args.output_folder.parent) as temporary:
            results = Path(temporary) / "results"
            cleaned = results / "msa_clean"
            pairs = results / "msa_dictionary"
            cleaned.mkdir(parents=True)
            pairs.mkdir()

            train_file = cleaned / "MSA_TRAIN_clean.txt"
            test_file = cleaned / "MSA_TEST_clean.txt"
            train_stats, train_set = clean_split(args.input_folder / "MSA_TRAIN.txt", train_file)
            test_stats, test_set = clean_split(args.input_folder / "MSA_TEST.txt", test_file)
            cleaning_report = {
                "MSA_TRAIN": train_stats,
                "MSA_TEST": test_stats,
                "exact_overlap_after_cleaning": len(train_set & test_set),
            }
            (cleaned / "cleaning_report.json").write_text(
                json.dumps(cleaning_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            train = train_file.read_text(encoding="utf-8").splitlines()
            build_dictionary_pairs(train, test_set, pairs)
            total = expand_pairs(train, test_set, pairs)
            results.rename(args.output_folder)
    except (OSError, UnicodeError) as error:
        parser.exit(1, f"Could not build the MSA dataset: {error}\n")

    final_file = args.output_folder / "msa_dictionary" / "train_pairs_50k.jsonl"
    print(f"Saved {total:,} training pairs to {final_file}")


if __name__ == "__main__":
    main()