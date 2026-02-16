#!/usr/bin/env python3
"""Generate the poetry corpus for Melometis Phase 1 planning-site analysis.

Produces 780 samples (260 x 3 categories):
  - Category A (rhyming): priming couplet + target couplet where line 4 rhymes with line 3
  - Category B (non_rhyming): same priming + line 3, but line 4 does NOT rhyme
  - Category C (generation): same priming + line 3 + trailing newline (no line 4)

Samples use character-based positions compatible with PLIP-rs positioning module.

Usage:
    python scripts/generate_poetry_corpus.py
    python scripts/generate_poetry_corpus.py --output corpus/attention_samples_poetry.json
    python scripts/generate_poetry_corpus.py --stats
    python scripts/generate_poetry_corpus.py --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import nltk

# ---------------------------------------------------------------------------
# CMU Pronouncing Dictionary
# The corpus/cmudict.dict file is not checked in (130K lines). Download from:
#   https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
# ---------------------------------------------------------------------------


def ensure_cmudict():
    """Download CMU dict if not already present."""
    try:
        nltk.corpus.cmudict.dict()
    except LookupError:
        print("Downloading CMU Pronouncing Dictionary...")
        nltk.download("cmudict", quiet=True)


def get_rhyme_suffix(phonemes):
    """Extract rhyme suffix: phonemes from last stressed vowel onward."""
    last_stressed = -1
    for i, p in enumerate(phonemes):
        if p[-1] == "1":  # primary stress marker
            last_stressed = i
    if last_stressed == -1:
        # No primary stress — try secondary stress
        for i, p in enumerate(phonemes):
            if p[-1] == "2":
                last_stressed = i
    if last_stressed == -1:
        # No stress at all — use the last vowel
        for i, p in enumerate(phonemes):
            if p[-1] in "012":
                last_stressed = i
    if last_stressed == -1:
        return None
    return tuple(phonemes[last_stressed:])


def build_rhyme_groups(cmu):
    """Group CMU dictionary words by rhyme suffix."""
    groups = {}
    for word, pronunciations in cmu.items():
        # Use first pronunciation only
        suffix = get_rhyme_suffix(pronunciations[0])
        if suffix is None:
            continue
        if suffix not in groups:
            groups[suffix] = []
        groups[suffix].append(word)
    return groups


# ---------------------------------------------------------------------------
# 20 curated rhyme groups with tagged words
# ---------------------------------------------------------------------------

# Each entry: (group_name, cmu_suffix_key, [(word, pos), ...])
# POS tags: "noun", "verb", "adj" — based on common end-of-line usage
# We'll validate these against CMU dict at runtime

RHYME_GROUPS = [
    # Strong families (Phase 0a)
    ("all", ("AO1", "L"), [
        ("call", "noun"), ("fall", "noun"), ("wall", "noun"), ("hall", "noun"),
        ("ball", "noun"), ("tall", "adj"), ("small", "adj"), ("stall", "noun"),
        ("crawl", "verb"), ("shawl", "noun"),
    ]),
    ("ound", ("AW1", "N", "D"), [
        ("ground", "noun"), ("sound", "noun"), ("found", "verb"), ("round", "adj"),
        ("mound", "noun"), ("hound", "noun"), ("bound", "adj"), ("wound", "noun"),
        ("drowned", "verb"), ("crowned", "verb"),
    ]),
    ("ight", ("AY1", "T"), [
        ("night", "noun"), ("light", "noun"), ("sight", "noun"), ("flight", "noun"),
        ("might", "noun"), ("bright", "adj"), ("right", "adj"), ("delight", "noun"),
        ("height", "noun"), ("kite", "noun"),
    ]),
    ("ee", ("IY1",), [
        ("tree", "noun"), ("sea", "noun"), ("free", "adj"), ("key", "noun"),
        ("knee", "noun"), ("bee", "noun"), ("plea", "noun"), ("flea", "noun"),
        ("spree", "noun"), ("decree", "noun"),
    ]),
    # Weak families (Phase 0a)
    ("oon", ("UW1", "N"), [
        ("moon", "noun"), ("noon", "noun"), ("spoon", "noun"), ("tune", "noun"),
        ("dune", "noun"), ("boon", "noun"), ("swoon", "verb"), ("lagoon", "noun"),
        ("balloon", "noun"), ("maroon", "noun"),
    ]),
    ("ake", ("EY1", "K"), [
        ("lake", "noun"), ("cake", "noun"), ("wake", "verb"), ("snake", "noun"),
        ("stake", "noun"), ("flake", "noun"), ("break", "verb"), ("shake", "verb"),
        ("drake", "noun"), ("mistake", "noun"),
    ]),
    ("ine", ("AY1", "N"), [
        ("wine", "noun"), ("line", "noun"), ("mine", "noun"), ("pine", "noun"),
        ("vine", "noun"), ("shine", "verb"), ("sign", "noun"), ("fine", "adj"),
        ("divine", "adj"), ("decline", "noun"),
    ]),
    # Additional groups for diversity
    ("ow", ("OW1",), [
        ("snow", "noun"), ("flow", "noun"), ("glow", "noun"), ("grow", "verb"),
        ("show", "noun"), ("know", "verb"), ("crow", "noun"), ("row", "noun"),
        ("bow", "noun"), ("shadow", "noun"),
    ]),
    ("ay", ("EY1",), [
        ("day", "noun"), ("way", "noun"), ("play", "noun"), ("stay", "verb"),
        ("say", "verb"), ("ray", "noun"), ("gray", "adj"), ("pray", "verb"),
        ("bay", "noun"), ("display", "noun"),
    ]),
    ("ore", ("AO1", "R"), [
        ("shore", "noun"), ("door", "noun"), ("floor", "noun"), ("store", "noun"),
        ("core", "noun"), ("roar", "noun"), ("soar", "verb"), ("pour", "verb"),
        ("explore", "verb"), ("restore", "verb"),
    ]),
    ("est", ("EH1", "S", "T"), [
        ("rest", "noun"), ("nest", "noun"), ("best", "adj"), ("west", "noun"),
        ("chest", "noun"), ("quest", "noun"), ("test", "noun"), ("guest", "noun"),
        ("crest", "noun"), ("request", "noun"),
    ]),
    ("own", ("AW1", "N"), [
        ("town", "noun"), ("crown", "noun"), ("gown", "noun"), ("brown", "adj"),
        ("frown", "noun"), ("down", "noun"), ("renown", "noun"), ("clown", "noun"),
        ("drown", "verb"), ("sundown", "noun"),
    ]),
    ("ame", ("EY1", "M"), [
        ("name", "noun"), ("flame", "noun"), ("game", "noun"), ("frame", "noun"),
        ("shame", "noun"), ("blame", "noun"), ("fame", "noun"), ("claim", "noun"),
        ("aim", "noun"), ("proclaim", "verb"),
    ]),
    ("old", ("OW1", "L", "D"), [
        ("gold", "noun"), ("cold", "adj"), ("bold", "adj"), ("fold", "noun"),
        ("told", "verb"), ("hold", "verb"), ("sold", "verb"), ("mold", "noun"),
        ("behold", "verb"), ("unfold", "verb"),
    ]),
    ("air", ("EH1", "R"), [
        ("chair", "noun"), ("stair", "noun"), ("hair", "noun"), ("fair", "adj"),
        ("pair", "noun"), ("care", "noun"), ("dare", "verb"), ("rare", "adj"),
        ("share", "noun"), ("affair", "noun"),
    ]),
    ("ing", ("IH1", "NG"), [
        ("king", "noun"), ("ring", "noun"), ("spring", "noun"), ("sing", "verb"),
        ("wing", "noun"), ("thing", "noun"), ("string", "noun"), ("swing", "noun"),
        ("bring", "verb"), ("offering", "noun"),
    ]),
    ("and", ("AE1", "N", "D"), [
        ("hand", "noun"), ("land", "noun"), ("sand", "noun"), ("band", "noun"),
        ("stand", "verb"), ("grand", "adj"), ("command", "noun"), ("demand", "noun"),
        ("expand", "verb"), ("understand", "verb"),
    ]),
    ("orn", ("AO1", "R", "N"), [
        ("morn", "noun"), ("born", "adj"), ("horn", "noun"), ("torn", "adj"),
        ("worn", "adj"), ("scorn", "noun"), ("sworn", "adj"), ("thorn", "noun"),
        ("forlorn", "adj"), ("adorn", "verb"),
    ]),
    ("ire", ("AY1", "ER0"), [
        ("fire", "noun"), ("wire", "noun"), ("desire", "noun"), ("tire", "verb"),
        ("hire", "verb"), ("choir", "noun"), ("inspire", "verb"), ("admire", "verb"),
        ("spire", "noun"), ("entire", "adj"),
    ]),
    ("ove", ("AH1", "V"), [
        ("love", "noun"), ("dove", "noun"), ("above", "noun"), ("glove", "noun"),
        ("shove", "verb"), ("foxglove", "noun"),
    ]),
]


# ---------------------------------------------------------------------------
# ~50 priming couplets (hand-crafted, labeled by rhyme group)
# ---------------------------------------------------------------------------

# Each: (rhyme_group_label, line1, line2)
# Lines do NOT end with punctuation that would interfere with position calculation
# The priming rhyme group must differ from the target rhyme group (enforced at assembly)

PRIMING_COUPLETS = [
    # -own group
    ("own", "The sun goes up, the sun goes down",
            "The moon shines bright above the town"),
    ("own", "The shepherd walked through fields of brown",
            "And led his flock beyond the town"),
    ("own", "A jester danced without a frown",
            "And entertained the king in town"),
    # -ue/-ew group
    ("ew", "The morning sky was clear and blue",
           "The birds sang out a song they knew"),
    ("ew", "The winter frost left morning dew",
           "On every leaf the cold wind blew"),
    ("ew", "A painter mixed each brilliant hue",
           "And captured scenes the whole world knew"),
    # -ight group
    ("ight", "The stars above were shining bright",
             "They lit the path throughout the night"),
    ("ight", "A falcon soared at breathless height",
             "Then vanished quickly out of sight"),
    ("ight", "The candle flickered warm and bright",
             "And cast its glow into the night"),
    # -ay group
    ("ay", "The flowers bloomed on that fine day",
           "And children gathered there to play"),
    ("ay", "The travelers set out on their way",
           "Before the dawn could start the day"),
    ("ay", "A wandering bard began to say",
           "A tale of kingdoms far away"),
    # -ee group
    ("ee", "The robin perched upon a tree",
           "And sang its song so wild and free"),
    ("ee", "The sailor gazed across the sea",
           "And longed for home and family"),
    ("ee", "A locksmith forged a golden key",
           "To open doors for all to see"),
    # -ore group
    ("ore", "The waves came crashing on the shore",
            "A sound the village folk adore"),
    ("ore", "The merchant opened up his store",
            "And spread his wares across the floor"),
    ("ore", "The thunder rumbled, then did roar",
            "And rain came falling more and more"),
    # -ound group
    ("ound", "The river flows without a sound",
             "Through meadows green and fertile ground"),
    ("ound", "The hunter tracked his faithful hound",
             "Across the hills and mossy ground"),
    ("ound", "A treasure lost was finally found",
             "Beneath the old familiar ground"),
    # -ine group
    ("ine", "The moonlight made the silver shine",
            "Through branches of the ancient pine"),
    ("ine", "The poet penned a graceful line",
            "With words as sweet as summer wine"),
    ("ine", "The morning brought a golden shine",
            "That warmed the frost on every vine"),
    # -ame group
    ("ame", "A hero earned his lasting fame",
            "By deeds that brought no hint of shame"),
    ("ame", "The blacksmith forged with steady aim",
            "And sparks flew high from every flame"),
    # -air group
    ("air", "The queen was known for beauty rare",
            "With flowers woven in her hair"),
    ("air", "A maiden sat upon a chair",
            "And gazed into the evening air"),
    # -est group
    ("est", "The weary traveler sought his rest",
            "Beyond the mountains to the west"),
    ("est", "The knight completed every quest",
            "And proved himself above the rest"),
    # -ing group
    ("ing", "A songbird perched upon a swing",
            "And warbled in the early spring"),
    ("ing", "The chapel bell began to ring",
            "And everyone began to sing"),
    # -all group
    ("all", "The ivy climbed along the wall",
            "And reached the tower proud and tall"),
    ("all", "The children gathered in the hall",
            "And played together with a ball"),
    # -old group
    ("old", "A story from the days of old",
            "Of daring knights and hearts of gold"),
    ("old", "The winter wind was sharp and cold",
            "A tale of hardship to be told"),
    # -and group
    ("and", "The castle stood upon the land",
            "A fortress built by careful hand"),
    ("and", "The desert stretched like golden sand",
            "As far as anyone could stand"),
    # -oon group
    ("oon", "The owl was hooting at the moon",
            "Its cry a melancholy tune"),
    ("oon", "The summer ended far too soon",
            "Beneath the harvest moon of June"),
    # -ake group
    ("ake", "The fisherman cast in the lake",
            "And waited there for morning's sake"),
    ("ake", "She baked a most delicious cake",
            "Enough for everyone to take"),
    # -ow group
    ("ow", "The garden sparkled in the snow",
           "And lanterns gave a gentle glow"),
    ("ow", "The river wound its way below",
           "With currents steady, smooth, and slow"),
    # -orn group
    ("orn", "The rooster crowed to greet the morn",
            "His cry rang out across the corn"),
    ("orn", "A uniform was creased and worn",
            "Its fabric faded, slightly torn"),
    # -ire group
    ("ire", "The villagers sat by the fire",
            "And watched the flames go ever higher"),
    ("ire", "A steeple topped with gleaming spire",
            "Rose up above to great inspire"),
    # -ove group
    ("ove", "A letter sealed with words of love",
            "Was carried by a snow white dove"),
    ("ove", "The sky stretched endlessly above",
            "As gentle as a mourning dove"),
]


# ---------------------------------------------------------------------------
# Line templates for target couplets
# ---------------------------------------------------------------------------

# Line-3 templates (the line whose ending word is the rhyme target)
# (pos, template_string) — {W} is the slot for the ending word
LINE3_TEMPLATES = [
    # Noun-ending templates
    ("noun", "The children played beside the {W}"),
    ("noun", "She walked alone beneath the {W}"),
    ("noun", "A gentle breeze came through the {W}"),
    ("noun", "He searched for hours near the {W}"),
    ("noun", "The old man rested by the {W}"),
    ("noun", "They gathered flowers near the {W}"),
    ("noun", "A bird took shelter in the {W}"),
    ("noun", "The traveler stopped beside the {W}"),
    ("noun", "She kept a secret like a {W}"),
    ("noun", "The painter captured every {W}"),
    ("noun", "A single candle lit the {W}"),
    ("noun", "The sailors ventured past the {W}"),
    ("noun", "He found his answer in the {W}"),
    ("noun", "The farmer worked beneath the {W}"),
    ("noun", "A whisper echoed through the {W}"),
    ("noun", "The poet wrote about the {W}"),
    ("noun", "She found her courage in the {W}"),
    ("noun", "The river flowed beside the {W}"),
    ("noun", "He built a cabin near the {W}"),
    ("noun", "The dancer moved across the {W}"),
    ("noun", "A stranger carried home a {W}"),
    ("noun", "The teacher pointed to the {W}"),
    # Verb-ending templates
    ("verb", "He tried his very best to {W}"),
    ("verb", "She closed her eyes and tried to {W}"),
    ("verb", "She taught the little ones to {W}"),
    # Adjective-ending templates
    ("adj", "The evening sky appeared so {W}"),
    ("adj", "The ancient castle stood so {W}"),
    ("adj", "The hidden valley seemed so {W}"),
]

# Line-4 templates (the line whose ending word rhymes or doesn't rhyme with line 3)
LINE4_TEMPLATES = [
    # Noun-ending templates
    ("noun", "And left behind a lonely {W}"),
    ("noun", "While dreaming of a distant {W}"),
    ("noun", "And waited for the distant {W}"),
    ("noun", "Where nothing stirred except the {W}"),
    ("noun", "And watched the sunset on the {W}"),
    ("noun", "To find at last the hidden {W}"),
    ("noun", "And listened to the evening {W}"),
    ("noun", "The silence broken by the {W}"),
    ("noun", "That echoed softly through the {W}"),
    ("noun", "And felt the presence of the {W}"),
    ("noun", "Where blossoms grew beside the {W}"),
    ("noun", "And thought about the coming {W}"),
    ("noun", "The moonlight shining on the {W}"),
    ("noun", "And drifted off to peaceful {W}"),
    ("noun", "The shadows fell across the {W}"),
    ("noun", "And heard the echoes of the {W}"),
    ("noun", "While standing near the ancient {W}"),
    ("noun", "And placed it gently on the {W}"),
    ("noun", "A memory of every {W}"),
    ("noun", "That glimmered in the fading {W}"),
    ("noun", "And wandered through the quiet {W}"),
    ("noun", "Beneath the ever-changing {W}"),
    # Verb-ending templates
    ("verb", "The silence made her want to {W}"),
    ("verb", "And everyone began to {W}"),
    ("verb", "The moment made the whole world {W}"),
    # Adjective-ending templates
    ("adj", "And everything appeared so {W}"),
    ("adj", "The path ahead was truly {W}"),
    ("adj", "The world around seemed very {W}"),
]


# ---------------------------------------------------------------------------
# Non-rhyming replacement words (for Category B)
# ---------------------------------------------------------------------------

# Common words that do NOT belong to any of the 20 selected rhyme groups.
# Tagged with POS for template compatibility.
NON_RHYMING_NOUNS = [
    "morning", "table", "river", "window", "garden", "mountain", "pillow",
    "village", "blanket", "feather", "rabbit", "captain", "forest",
    "meadow", "harbor", "lantern", "shadow", "whisper", "basket",
    "silence", "ember", "mirror", "candle", "saddle", "timber",
    "crystal", "thunder", "cipher", "blossom", "amber",
]
NON_RHYMING_VERBS = [
    "wander", "stumble", "whisper", "gather", "linger", "tremble",
    "shatter", "ponder", "murmur", "fumble",
]
NON_RHYMING_ADJS = [
    "bitter", "gentle", "silver", "purple", "golden", "crimson",
    "ancient", "tender", "solemn", "humble",
]


# ---------------------------------------------------------------------------
# Assembly logic
# ---------------------------------------------------------------------------


def get_words_by_pos(group_entry, pos):
    """Get all words in a rhyme group matching a given POS tag."""
    _, _, words = group_entry
    return [w for w, p in words if p == pos]


def get_all_word_pairs(group_entry):
    """Generate all unique ordered pairs of words from a rhyme group."""
    _, _, words = group_entry
    pairs = []
    word_list = [w for w, _ in words]
    for i in range(len(word_list)):
        for j in range(len(word_list)):
            if i != j:
                pairs.append((word_list[i], words[i][1], word_list[j], words[j][1]))
    return pairs  # (w1, pos1, w2, pos2)


def get_compatible_templates(templates, pos):
    """Get templates compatible with a given POS."""
    return [(p, t) for p, t in templates if p == pos]


def pick_non_rhyming_word(pos, rhyme_suffixes_to_avoid, cmu, rng):
    """Pick a non-rhyming replacement word matching the given POS."""
    if pos == "noun":
        pool = NON_RHYMING_NOUNS
    elif pos == "verb":
        pool = NON_RHYMING_VERBS
    elif pos == "adj":
        pool = NON_RHYMING_ADJS
    else:
        pool = NON_RHYMING_NOUNS  # fallback

    candidates = []
    for word in pool:
        # Check it doesn't accidentally rhyme with any of our target groups
        if word.lower() in cmu:
            suffix = get_rhyme_suffix(cmu[word.lower()][0])
            if suffix not in rhyme_suffixes_to_avoid:
                candidates.append(word)
        else:
            # Word not in CMU — unlikely to rhyme
            candidates.append(word)

    if not candidates:
        # Fallback: just pick from pool
        candidates = pool

    return rng.choice(candidates)


def compute_positions(code, ending_word):
    """Compute marker_char_pos and target_char_positions for a sample.

    marker_char_pos: position of the \\n between line 3 and line 4
                     (or the trailing \\n for Category C)
    target_char_positions: character positions of ending_word in line 3
    """
    lines = code.split("\n")
    # marker is the newline after line 3 (0-indexed: lines[2])
    # Position = sum of lengths of lines[0..2] + 2 newlines
    marker_pos = len(lines[0]) + 1 + len(lines[1]) + 1 + len(lines[2])

    # The ending word is at the end of line 3 (lines[2])
    word_len = len(ending_word)
    word_start = marker_pos - word_len
    target_positions = list(range(word_start, marker_pos))

    return marker_pos, target_positions


def generate_corpus(seed=42):
    """Generate all 780 samples."""
    ensure_cmudict()
    cmu = nltk.corpus.cmudict.dict()
    rng = random.Random(seed)

    # Collect all rhyme suffixes to avoid for non-rhyming words
    all_rhyme_suffixes = set()
    for _, suffix, _ in RHYME_GROUPS:
        all_rhyme_suffixes.add(suffix)

    # Build priming couplet rotation (avoid confounding)
    priming_pool = list(PRIMING_COUPLETS)

    samples_a = []  # Category A: rhyming
    samples_b = []  # Category B: non-rhyming
    samples_c = []  # Category C: generation

    triplet_id = 0

    for group_idx, group_entry in enumerate(RHYME_GROUPS):
        group_name, group_suffix, group_words = group_entry
        n_samples_for_group = 13

        # Get all word pairs and compatible templates
        all_pairs = get_all_word_pairs(group_entry)
        rng.shuffle(all_pairs)

        # Filter priming couplets that DON'T have the same rhyme group
        available_primings = [
            (label, l1, l2) for label, l1, l2 in priming_pool
            if label != group_name
        ]
        if not available_primings:
            print(f"WARNING: No priming couplets available for group '{group_name}'")
            available_primings = priming_pool  # fallback

        generated = 0
        pair_idx = 0
        template_pairs_used = set()

        while generated < n_samples_for_group and pair_idx < len(all_pairs):
            w1, pos1, w2, pos2 = all_pairs[pair_idx]
            pair_idx += 1

            # Find compatible templates for w1 (line 3) and w2 (line 4)
            t3_candidates = get_compatible_templates(LINE3_TEMPLATES, pos1)
            t4_candidates = get_compatible_templates(LINE4_TEMPLATES, pos2)

            if not t3_candidates or not t4_candidates:
                continue

            # Pick template combination not yet used in this group
            found = False
            rng.shuffle(t3_candidates)
            rng.shuffle(t4_candidates)
            for t3_pos, t3 in t3_candidates:
                for t4_pos, t4 in t4_candidates:
                    key = (t3, t4)
                    if key not in template_pairs_used:
                        template_pairs_used.add(key)
                        found = True
                        break
                if found:
                    break

            if not found:
                # All template combos used — allow repeats
                t3_pos, t3 = rng.choice(t3_candidates)
                t4_pos, t4 = rng.choice(t4_candidates)

            # Pick priming couplet (round-robin, different group)
            priming_idx = (triplet_id) % len(available_primings)
            priming_label, priming_l1, priming_l2 = available_primings[priming_idx]

            # Build line 3 and line 4
            line3 = t3.replace("{W}", w1)
            line4_rhyming = t4.replace("{W}", w2)

            # --- Category A: rhyming ---
            code_a = f"{priming_l1}\n{priming_l2}\n{line3}\n{line4_rhyming}"
            marker_pos, target_positions = compute_positions(code_a, w1)

            sample_a = {
                "id": f"couplet_{triplet_id:03d}",
                "code": code_a,
                "priming_lines": 2,
                "marker_char_pos": marker_pos,
                "marker_pattern": "\n",
                "target_char_positions": target_positions,
                "rhyme_group": group_name,
                "ending_word": w1,
                "rhyme_word": w2,
                "category": "A",
                "triplet_id": triplet_id,
            }
            samples_a.append(sample_a)

            # --- Category B: non-rhyming control ---
            non_rhyme_word = pick_non_rhyming_word(
                pos2, all_rhyme_suffixes, cmu, rng
            )
            line4_non_rhyming = t4.replace("{W}", non_rhyme_word)
            code_b = f"{priming_l1}\n{priming_l2}\n{line3}\n{line4_non_rhyming}"
            # marker_pos and target_positions are the same (same priming + line 3)

            sample_b = {
                "id": f"prose_{triplet_id:03d}",
                "code": code_b,
                "priming_lines": 2,
                "marker_char_pos": marker_pos,
                "marker_pattern": "\n",
                "target_char_positions": target_positions,
                "rhyme_group": group_name,
                "ending_word": w1,
                "rhyme_word": None,
                "category": "B",
                "triplet_id": triplet_id,
            }
            samples_b.append(sample_b)

            # --- Category C: generation setup (truncated) ---
            code_c = f"{priming_l1}\n{priming_l2}\n{line3}\n"
            # For Category C, marker_char_pos is the trailing \n (same position)

            sample_c = {
                "id": f"gen_{triplet_id:03d}",
                "code": code_c,
                "priming_lines": 2,
                "marker_char_pos": marker_pos,
                "marker_pattern": "\n",
                "target_char_positions": target_positions,
                "rhyme_group": group_name,
                "ending_word": w1,
                "rhyme_word": None,
                "category": "C",
                "triplet_id": triplet_id,
            }
            samples_c.append(sample_c)

            generated += 1
            triplet_id += 1

        if generated < n_samples_for_group:
            print(
                f"WARNING: group '{group_name}' only produced {generated}/{n_samples_for_group} samples"
            )

    return samples_a, samples_b, samples_c


def print_stats(samples_a, samples_b, samples_c):
    """Print corpus statistics."""
    total = len(samples_a) + len(samples_b) + len(samples_c)
    print(f"\n=== Poetry Corpus Statistics ===")
    print(f"Total samples: {total}")
    print(f"  Category A (rhyming):     {len(samples_a)}")
    print(f"  Category B (non-rhyming): {len(samples_b)}")
    print(f"  Category C (generation):  {len(samples_c)}")

    # Rhyme group distribution
    from collections import Counter
    groups = Counter(s["rhyme_group"] for s in samples_a)
    print(f"\nRhyme group distribution (Category A):")
    for group, count in sorted(groups.items()):
        print(f"  {group:>8}: {count}")

    # Priming confound check
    confounded = 0
    for s in samples_a:
        code = s["code"]
        lines = code.split("\n")
        # Check if priming couplet shares rhyme group with target
        # (This is a simplified check — just report the group)
        pass

    # Sample lengths
    lengths = [len(s["code"]) for s in samples_a]
    print(f"\nCategory A code lengths: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.0f}")

    # Show a few examples
    print(f"\n=== Sample Examples ===")
    for i, (a, b, c) in enumerate(zip(samples_a[:3], samples_b[:3], samples_c[:3])):
        print(f"\n--- Triplet {a['triplet_id']} (group: {a['rhyme_group']}) ---")
        print(f"Cat A (rhyming):     {a['ending_word']} / {a['rhyme_word']}")
        print(f"  {a['code']!r}")
        print(f"  marker_char_pos={a['marker_char_pos']}, "
              f"target_char_positions={a['target_char_positions']}")
        print(f"Cat B (non-rhyming): {b['ending_word']} / (no rhyme)")
        print(f"  {b['code']!r}")
        print(f"Cat C (generation):")
        print(f"  {c['code']!r}")


def write_corpus(samples_a, samples_b, samples_c, output_path):
    """Write corpus to JSON file."""
    corpus = {
        "_format_version": "2.0",
        "_description": "Poetry corpus for planning-site analysis (Melometis Phase 1)",
        "_n_samples": len(samples_a) + len(samples_b) + len(samples_c),
        "_n_per_category": len(samples_a),
        "_n_rhyme_groups": len(set(s["rhyme_group"] for s in samples_a)),
        "rhyming": samples_a,
        "non_rhyming": samples_b,
        "generation": samples_c,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    size_kb = output.stat().st_size / 1024
    print(f"\nCorpus written to: {output}")
    print(f"File size: {size_kb:.1f} KB")


def validate_positions(samples):
    """Quick validation of character positions."""
    errors = 0
    for s in samples:
        code = s["code"]
        marker = s["marker_char_pos"]

        # Check marker points to newline
        if marker >= len(code):
            if s["category"] != "C" or marker != len(code):
                print(f"ERROR {s['id']}: marker_char_pos={marker} >= len(code)={len(code)}")
                errors += 1
                continue
        elif code[marker] != "\n":
            print(f"ERROR {s['id']}: code[{marker}] = {code[marker]!r}, expected '\\n'")
            errors += 1
            continue

        # Check target positions
        for pos in s["target_char_positions"]:
            if pos >= len(code):
                print(f"ERROR {s['id']}: target_char_pos={pos} >= len(code)={len(code)}")
                errors += 1
            elif code[pos] == "\n" or code[pos] == " ":
                print(f"ERROR {s['id']}: target char at {pos} is whitespace: {code[pos]!r}")
                errors += 1

        # Check target positions form the ending word
        if s["target_char_positions"]:
            extracted = code[s["target_char_positions"][0]:s["target_char_positions"][-1]+1]
            if extracted != s["ending_word"]:
                print(f"ERROR {s['id']}: extracted '{extracted}' != ending_word '{s['ending_word']}'")
                errors += 1

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate poetry corpus for Melometis Phase 1")
    parser.add_argument(
        "--output", default="corpus/attention_samples_poetry.json",
        help="Output JSON path (default: corpus/attention_samples_poetry.json)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    print("Generating poetry corpus...")
    samples_a, samples_b, samples_c = generate_corpus(seed=args.seed)

    # Validate positions
    all_samples = samples_a + samples_b + samples_c
    errors = validate_positions(all_samples)
    if errors > 0:
        print(f"\n*** {errors} position errors found! ***")
        sys.exit(1)
    else:
        print(f"\nPosition validation: {len(all_samples)} samples, 0 errors")

    print_stats(samples_a, samples_b, samples_c)

    if not args.stats:
        write_corpus(samples_a, samples_b, samples_c, args.output)


if __name__ == "__main__":
    main()
