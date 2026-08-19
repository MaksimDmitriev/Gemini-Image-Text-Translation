#!/usr/bin/env python3
"""Translate Russian labels in a technical drawing with Gemini."""

import argparse
import io
import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFont


PROMPT = """
Find every text fragment containing Cyrillic letters in this technical drawing.
Translate each fragment from Russian to English. Transliterate abbreviations instead
of translating them (for example, Russian abbreviation "ТУ" becomes "TU"). Do not
return standalone numbers or text already written only in Latin characters.

Return ONLY a JSON array. Each object must have exactly these fields:
  "source": the original Russian text,
  "translation": its English translation,
  "box_2d": tight bounds of the original text [ymin, xmin, ymax, xmax],
  "target_box_2d": the largest safe rectangular area in the same table cell or label
                   area where the translation may be written without crossing lines.

All coordinates must be integers normalized to 0..1000. Include all Cyrillic text,
including small labels, notes, title blocks, and rotated text. If text is rotated,
still use its axis-aligned bounds. Preserve numbers that occur inside translated text.
"""

FALLBACK_MODEL = "gemini-3.5-flash-lite"
RECHECK_PROMPT = (
    "This drawing has already been partially translated. Find and translate only "
    "Cyrillic text that still remains; ignore all English replacements.\n\n" + PROMPT
)


def normalized_box(box, width, height):
    """Convert Gemini's [y1, x1, y2, x2] coordinates to a PIL rectangle."""
    y1, x1, y2, x2 = (max(0, min(1000, int(value))) for value in box)
    return (
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    )


def parse_regions(raw):
    """Accept plain JSON and tolerate an accidental Markdown code fence."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Gemini did not return a JSON array:\n{raw}")
    regions = json.loads(match.group())
    required = {"source", "translation", "box_2d", "target_box_2d"}
    return [item for item in regions if required <= item.keys()]


def font_at(size):
    """Use a commonly available readable font, with Pillow's bundled fallback."""
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    for word in words:
        candidate = word if not lines else f"{lines[-1]} {word}"
        if not lines or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            if lines:
                lines[-1] = candidate
            else:
                lines.append(candidate)
        else:
            lines.append(word)
    return "\n".join(lines)


def fitted_text(draw, text, box, max_font_size):
    """Choose the largest font and wrapping that remains inside the target box."""
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    start_size = min(max(6, height), max(6, round(max_font_size)))
    for size in range(start_size, 5, -1):
        font = font_at(size)
        wrapped = wrap_text(draw, text, font, width)
        bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=1)
        if bounds[2] - bounds[0] <= width and bounds[3] - bounds[1] <= height:
            return font, wrapped, bounds
    font = font_at(6)
    wrapped = wrap_text(draw, text, font, width)
    return font, wrapped, draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=1)


def draw_fitted_text(image, draw, text, box, clockwise_rotation, max_font_size):
    """Fit text to a cell, rotate it like the source, and paste it within the cell."""
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    padding = max(1, round(min(width, height) * 0.04))
    padding = min(padding, (width - 1) // 2, (height - 1) // 2)
    x1, y1, x2, y2 = x1 + padding, y1 + padding, x2 - padding, y2 - padding
    width, height = max(1, x2 - x1), max(1, y2 - y1)

    try:
        rotation = (round(float(clockwise_rotation) / 90) * 90) % 360
    except (TypeError, ValueError):
        rotation = 0
    layout_width, layout_height = (
        (height, width) if rotation in (90, 270) else (width, height)
    )
    layout_box = (0, 0, layout_width, layout_height)
    font, wrapped, bounds = fitted_text(draw, text, layout_box, max_font_size)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]

    layer = Image.new("RGBA", (layout_width, layout_height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    position = (
        (layout_width - text_width) / 2 - bounds[0],
        (layout_height - text_height) / 2 - bounds[1],
    )
    layer_draw.multiline_text(
        position, wrapped, font=font, fill="black", spacing=1, align="center"
    )
    if rotation:
        layer = layer.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, (x1, y1), layer)


def request_regions(client, image_bytes, mime_type, model, prompt):
    """Request translated regions, falling back only when the model returns 503."""
    contents = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
    config = types.GenerateContentConfig(response_mime_type="application/json")
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except errors.ServerError as error:
        if error.code != 503 or model == FALLBACK_MODEL:
            raise
        print(f"{model} is unavailable; retrying with {FALLBACK_MODEL}")
        response = client.models.generate_content(
            model=FALLBACK_MODEL, contents=contents, config=config
        )
    return parse_regions(response.text)


def apply_regions(image, regions):
    """Erase and replace all regions, preserving the source text's approximate size."""
    draw = ImageDraw.Draw(image)
    for region in regions:
        erase_box = normalized_box(region["box_2d"], image.width, image.height)
        target_box = normalized_box(region["target_box_2d"], image.width, image.height)
        draw.rectangle(erase_box, fill="white")

        source_width = max(1, erase_box[2] - erase_box[0])
        source_height = max(1, erase_box[3] - erase_box[1])
        rotation = region.get("rotation_degrees", 0)
        try:
            is_vertical = round(float(rotation) / 90) % 2 == 1
        except (TypeError, ValueError):
            is_vertical = False
        source_text_height = source_width if is_vertical else source_height
        draw_fitted_text(
            image,
            draw,
            str(region["translation"]),
            target_box,
            rotation,
            max_font_size=source_text_height * 1.2,
        )


def translate_image(input_path, output_path, model):
    with Image.open(input_path) as source:
        mime_type = Image.MIME.get(source.format, "image/png")
        image = source.convert("RGB")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    regions = request_regions(
        client, Path(input_path).read_bytes(), mime_type, model, PROMPT
    )
    apply_regions(image, regions)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    remaining_regions = request_regions(
        client, buffer.getvalue(), "image/png", model, RECHECK_PROMPT
    )
    apply_regions(image, remaining_regions)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(
        f"Translated {len(regions) + len(remaining_regions)} text regions "
        f"({len(remaining_regions)} on second pass) -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Russian drawing (PNG/JPEG/WebP)")
    parser.add_argument("output", nargs="?", help="Output image path")
    parser.add_argument("--model", default="gemini-3.6-flash")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        parser.error("export GEMINI_API_KEY in your shell environment")
    source = Path(args.input)
    destination = Path(args.output) if args.output else source.with_name(
        f"{source.stem}_translated.png"
    )
    translate_image(source, destination, args.model)


if __name__ == "__main__":
    main()
