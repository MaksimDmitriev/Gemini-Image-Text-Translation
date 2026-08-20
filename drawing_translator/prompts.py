"""Prompts shared by model-backed providers."""

REGION_PROMPT = """
Find every text fragment containing Cyrillic letters in this technical drawing.
Translate each fragment from Russian to English. Transliterate only short
abbreviations, normally 2-5 letters that are uppercase or followed by a period (for
example, Russian abbreviation "ТУ" becomes "TU"). Translate every ordinary Russian
word or phrase; never transliterate a complete word or phrase. Do not return
standalone numbers or text already written only in Latin characters.

Return ONLY a JSON array. Each object must have exactly these fields:
  "source": the original Russian text,
  "translation": its English translation,
  "box_2d": tight bounds of the original text [ymin, xmin, ymax, xmax],
  "target_box_2d": the largest safe rectangular area in the same table cell or label
                   area where the translation may be written without crossing lines,
  "rotation_degrees": clockwise text rotation; use exactly 0, 90, 180, or 270.

All coordinates must be integers normalized to 0..1000. Include all Cyrillic text,
including small labels, notes, title blocks, and rotated text. If text is rotated,
still use its axis-aligned bounds. Preserve numbers that occur inside translated text.
"""

CLEANUP_REGION_PROMPT = (
    "This drawing has already been partially translated. Find and translate only "
    "Cyrillic text that still remains; ignore all English replacements.\n\n"
    + REGION_PROMPT
)

TRANSLATION_PROMPT = """
Translate each supplied Russian technical-drawing label into concise English.
Transliterate only short abbreviations, normally 2-5 letters that are uppercase or
followed by a period. Translate ordinary words and phrases rather than transliterating
them. Preserve numbers, standards, dimensions, and punctuation. Return one result for
every input id and do not add explanations.
"""
