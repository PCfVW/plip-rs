#!/usr/bin/env python3
"""Review poetry corpus for awkward template-word combinations.

Extracts every line-3 and line-4 ending, groups by template, and flags
combinations where the filled word likely clashes with the template context.

Usage:
    python scripts/review_poetry_corpus.py
    python scripts/review_poetry_corpus.py --corpus corpus/attention_samples_poetry.json
"""

import argparse
import json
import re
from collections import defaultdict


def extract_template_and_word(line, ending_word):
    """Split a line into its template context and the filled end-word."""
    if line.endswith(ending_word):
        context = line[: len(line) - len(ending_word)].rstrip()
        return context, ending_word
    return line, ""


# Words that are semantically restricted — they clash with many contexts
# Key: end-word, Value: list of preceding-word patterns that make it awkward
KNOWN_CLASHES = {
    # Adjectives used as line-ending words
    "small": ["warm", "dark", "immense"],
    "tall": ["warm", "dark"],
    "bright": ["dark"],
    "cold": ["golden", "brighter", "warm"],
    "bold": ["golden", "peaceful", "quiet", "fading", "silver"],
    "round": ["golden", "brighter", "peaceful"],
    "bound": ["golden", "brighter", "peaceful"],
    "brown": ["golden", "brighter", "silver"],
    "gray": ["brighter", "golden"],
    "fair": ["winter", "fading"],
    "rare": ["winter", "morning"],
    "fine": ["winter", "fading"],
    "born": ["golden", "brighter", "morning"],
    "torn": ["golden", "brighter", "morning"],
    "worn": ["golden", "brighter", "morning"],
    "sworn": ["golden", "brighter"],
    "entire": ["golden", "brighter"],
    "grand": ["winter", "fading", "silver"],
    # Nouns that clash with certain modifiers
    "ball": ["winter", "morning", "golden", "peaceful", "silver", "fading"],
    "wall": ["winter", "morning", "golden", "peaceful", "trail"],
    "hall": ["winter", "morning", "trail"],
    "stall": ["winter", "morning", "golden", "silver", "brighter"],
    "shawl": ["winter", "golden", "trail"],
    "crawl": ["golden", "silver", "morning"],
    "mound": ["winter", "morning", "golden", "silver", "brighter"],
    "hound": ["winter", "morning", "golden", "silver", "brighter", "trail"],
    "wound": ["winter", "morning", "golden", "silver", "brighter"],
    "kite": ["winter", "morning", "golden", "trail"],
    "flea": ["winter", "golden", "brighter", "silver", "trail"],
    "plea": ["winter", "golden", "silver"],
    "spree": ["winter", "golden", "silver"],
    "decree": ["winter", "golden", "silver"],
    "boon": ["winter", "golden", "silver", "trail"],
    "swoon": ["golden", "silver", "trail"],
    "lagoon": ["winter", "trail"],
    "maroon": ["winter", "trail", "silver"],
    "drake": ["winter", "golden", "silver", "trail"],
    "flake": ["golden", "silver", "trail"],
    "mistake": ["golden", "silver", "trail"],
    "stake": ["golden", "silver"],
    "decline": ["golden", "silver", "trail"],
    "scorn": ["golden", "silver", "brighter", "trail"],
    "thorn": ["golden", "brighter", "trail"],
    "choir": ["golden", "silver", "trail"],
    "spire": ["golden", "trail"],
    "glove": ["golden", "silver", "trail"],
    "shove": ["golden", "silver", "morning", "trail"],
    "foxglove": ["golden", "silver", "morning", "winter", "trail", "brighter"],
}


def check_awkward(line, ending_word):
    """Check if a line has an awkward template-word combination.

    Returns a list of (reason,) strings, empty if OK.
    """
    issues = []
    words = line.lower().split()
    if len(words) < 2:
        return issues

    # Get the 1-3 words preceding the end-word
    end_lower = ending_word.lower()
    preceding = words[:-1] if words[-1] == end_lower else words

    # Check known clashes
    if end_lower in KNOWN_CLASHES:
        for clash_word in KNOWN_CLASHES[end_lower]:
            if any(clash_word in w for w in preceding[-3:]):
                issues.append(f"clash: '{clash_word}...{ending_word}'")

    # Generic heuristic: adjective end-words after "and" often sound odd
    # if the preceding adjective has a different semantic field
    if len(preceding) >= 2 and preceding[-1] == "and":
        # "warm and small", "dark and bold", etc.
        issues.append(f"'...{preceding[-2]} and {ending_word}' — check semantic fit")

    # "of {noun}" where noun is abstract/doesn't fit
    if len(preceding) >= 1 and preceding[-1] == "of":
        # "trail of wall", "trail of stall" etc.
        abstract_nouns = {
            "wall", "stall", "hall", "ball", "shawl", "crawl",
            "mound", "hound", "wound", "kite", "flea", "boon",
            "drake", "flake", "stake", "scorn", "thorn", "glove",
            "shove", "foxglove", "swoon", "spire", "choir",
        }
        if end_lower in abstract_nouns:
            context_before_of = preceding[-2] if len(preceding) >= 2 else "?"
            issues.append(f"'...{context_before_of} of {ending_word}' — check noun fit")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Review poetry corpus for awkward combinations")
    parser.add_argument(
        "--corpus", default="corpus/attention_samples_poetry.json",
        help="Path to corpus JSON"
    )
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)

    rhyming = corpus["rhyming"]

    # Group samples by template context for line 3 and line 4
    line3_by_template = defaultdict(list)
    line4_by_template = defaultdict(list)
    all_issues = []

    for sample in rhyming:
        lines = sample["code"].split("\n")
        line3 = lines[2]
        line4 = lines[3]
        ending = sample["ending_word"]
        rhyme = sample.get("rhyme_word", "?")
        tid = sample["triplet_id"]
        group = sample["rhyme_group"]

        ctx3, w3 = extract_template_and_word(line3, ending)
        ctx4, w4 = extract_template_and_word(line4, rhyme) if rhyme else (line4, "")

        line3_by_template[ctx3].append((w3, group, tid))
        if ctx4:
            line4_by_template[ctx4].append((w4, group, tid))

        # Check for awkwardness
        issues3 = check_awkward(line3, ending)
        issues4 = check_awkward(line4, rhyme) if rhyme else []

        for issue in issues3:
            all_issues.append((tid, group, "line3", line3, issue))
        for issue in issues4:
            all_issues.append((tid, group, "line4", line4, issue))

    # Report
    print("=" * 70)
    print("POETRY CORPUS REVIEW — AWKWARD COMBINATIONS")
    print("=" * 70)

    if all_issues:
        print(f"\n### {len(all_issues)} potential issues found ###\n")

        # Group by line
        by_line = defaultdict(list)
        for tid, group, which_line, full_line, issue in all_issues:
            by_line[which_line].append((tid, group, full_line, issue))

        for which_line in ["line3", "line4"]:
            items = by_line.get(which_line, [])
            if items:
                print(f"\n--- {which_line.upper()} issues ({len(items)}) ---\n")
                for tid, group, full_line, issue in sorted(items, key=lambda x: x[0]):
                    print(f"  [{tid:03d}] ({group:>5})  {issue}")
                    print(f"         {full_line}")
                    print()
    else:
        print("\nNo issues detected.")

    # Summary stats
    print("=" * 70)
    print(f"Total Category A samples: {len(rhyming)}")
    print(f"Unique line-3 templates: {len(line3_by_template)}")
    print(f"Unique line-4 templates: {len(line4_by_template)}")
    print(f"Potential issues: {len(all_issues)}")
    print(f"  line3: {sum(1 for _, _, w, _, _ in all_issues if w == 'line3')}")
    print(f"  line4: {sum(1 for _, _, w, _, _ in all_issues if w == 'line4')}")


if __name__ == "__main__":
    main()
