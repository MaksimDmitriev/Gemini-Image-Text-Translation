"""Erase source labels and fit translated text back into a drawing."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from drawing_translator.models import Box, TextRegion


def normalized_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    y1, x1, y2, x2 = (max(0, min(1000, int(value))) for value in box)
    return (
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    )


def font_at(size: int):
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


def wrap_text(draw, text: str, font, max_width: int) -> str:
    lines: list[str] = []
    for word in text.split():
        candidate = word if not lines else f"{lines[-1]} {word}"
        if not lines or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            if lines:
                lines[-1] = candidate
            else:
                lines.append(candidate)
        else:
            lines.append(word)
    return "\n".join(lines)


def fitted_text(draw, text: str, width: int, height: int, max_font_size: float):
    start_size = min(max(6, height), max(6, round(max_font_size)))
    for size in range(start_size, 5, -1):
        font = font_at(size)
        wrapped = wrap_text(draw, text, font, width)
        bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=1)
        if bounds[2] - bounds[0] <= width and bounds[3] - bounds[1] <= height:
            return font, wrapped, bounds
    font = font_at(6)
    wrapped = wrap_text(draw, text, font, width)
    return font, wrapped, draw.multiline_textbbox(
        (0, 0), wrapped, font=font, spacing=1
    )


def normalized_rotation(value: object) -> int:
    try:
        return (round(float(value) / 90) * 90) % 360
    except (TypeError, ValueError):
        return 0


def _draw_fitted_text(
    image,
    draw,
    text: str,
    box: tuple[int, int, int, int],
    max_font_size: float,
    rotation: int,
):
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    padding = max(1, round(min(width, height) * 0.04))
    padding = min(padding, (width - 1) // 2, (height - 1) // 2)
    x1, y1, x2, y2 = x1 + padding, y1 + padding, x2 - padding, y2 - padding
    width, height = max(1, x2 - x1), max(1, y2 - y1)

    layout_width, layout_height = (
        (height, width) if rotation in (90, 270) else (width, height)
    )
    font, wrapped, bounds = fitted_text(
        draw, text, layout_width, layout_height, max_font_size
    )
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


def _line_masks(image):
    """Extract long horizontal and vertical lines for conservative cell expansion."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, image.width // 80), 1)
        ),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(15, image.height // 80))
        ),
    )
    return horizontal, vertical


def _cell_box(box: tuple[int, int, int, int], masks, image_size):
    if masks is None:
        return box
    import numpy as np

    horizontal, vertical = masks
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    search_x = max(40, width * 5)
    search_y = max(40, height * 5)
    horizontal_profile = np.count_nonzero(horizontal[:, x1:x2], axis=1)
    vertical_profile = np.count_nonzero(vertical[y1:y2, :], axis=0)

    def nearest(profile, start, direction, limit, threshold):
        stop = (
            max(0, start - limit)
            if direction < 0
            else min(len(profile) - 1, start + limit)
        )
        positions = range(start, stop, direction)
        return next((p for p in positions if profile[p] >= threshold), None)

    left = nearest(
        vertical_profile, max(0, x1 - 1), -1, search_x, max(2, height * 0.55)
    )
    right = nearest(
        vertical_profile,
        min(image_width - 1, x2 + 1),
        1,
        search_x,
        max(2, height * 0.55),
    )
    top = nearest(
        horizontal_profile, max(0, y1 - 1), -1, search_y, max(2, width * 0.55)
    )
    bottom = nearest(
        horizontal_profile,
        min(image_height - 1, y2 + 1),
        1,
        search_y,
        max(2, width * 0.55),
    )
    if None in (left, right, top, bottom):
        return box
    candidate = (left + 1, top + 1, right, bottom)
    if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
        return box

    # Use the cell's space only in the label's writing direction. Keeping a narrow
    # band in the other direction avoids covering unrelated values in the same cell.
    if width >= height:
        band_padding = max(1, height // 2)
        return (
            candidate[0],
            max(candidate[1], y1 - band_padding),
            candidate[2],
            min(candidate[3], y2 + band_padding),
        )
    band_padding = max(1, width // 2)
    return (
        max(candidate[0], x1 - band_padding),
        candidate[1],
        min(candidate[2], x2 + band_padding),
        candidate[3],
    )


def apply_regions(image: Image.Image, regions: list[TextRegion]) -> None:
    draw = ImageDraw.Draw(image)
    masks = _line_masks(image)
    for region in regions:
        erase_box = normalized_box(region.box_2d, image.width, image.height)
        requested_target = normalized_box(
            region.target_box_2d, image.width, image.height
        )
        target_box = (
            _cell_box(requested_target, masks, image.size)
            if requested_target == erase_box
            else requested_target
        )
        draw.rectangle(erase_box, fill="white")

        rotation = normalized_rotation(region.rotation_degrees)
        vertical = rotation in (90, 270)
        source_width = max(1, erase_box[2] - erase_box[0])
        source_height = max(1, erase_box[3] - erase_box[1])
        source_thickness = source_width if vertical else source_height
        _draw_fitted_text(
            image,
            draw,
            region.translation,
            target_box,
            source_thickness * 1.4,
            rotation,
        )
