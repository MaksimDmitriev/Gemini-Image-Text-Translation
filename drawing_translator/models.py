"""Shared data types and provider interface."""

from dataclasses import dataclass
from typing import Protocol


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TextRegion:
    """A translated image region using normalized [y1, x1, y2, x2] boxes."""

    source: str
    translation: str
    box_2d: Box
    target_box_2d: Box
    rotation_degrees: int = 0


class RegionProvider(Protocol):
    """Find Cyrillic text in an image and return translated regions."""

    name: str

    def find_regions(
        self, image_bytes: bytes, mime_type: str, *, cleanup: bool = False
    ) -> list[TextRegion]: ...
