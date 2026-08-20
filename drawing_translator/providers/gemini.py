"""Gemini vision implementation of the region-provider interface."""

import json
import os
import re

from google import genai
from google.genai import errors, types

from drawing_translator.models import Box, TextRegion
from drawing_translator.prompts import CLEANUP_REGION_PROMPT, REGION_PROMPT


def _box(value: object) -> Box:
    def numbers(item: object):
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from numbers(child)
        else:
            yield item

    values = list(numbers(value))
    if len(values) != 4:
        raise ValueError(f"invalid Gemini bounding box: {value!r}")
    return tuple(max(0, min(1000, int(number))) for number in values)  # type: ignore[return-value]


def parse_regions(raw: str) -> list[TextRegion]:
    """Accept plain JSON and tolerate an accidental Markdown code fence."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Gemini did not return a JSON array:\n{raw}")
    items = json.loads(match.group())
    required = {"source", "translation", "box_2d", "target_box_2d"}
    return [
        TextRegion(
            source=str(item["source"]),
            translation=str(item["translation"]),
            box_2d=_box(item["box_2d"]),
            target_box_2d=_box(item["target_box_2d"]),
            rotation_degrees=int(item.get("rotation_degrees", 0)),
        )
        for item in items
        if isinstance(item, dict) and required <= item.keys()
    ]


class GeminiProvider:
    name = "gemini"
    default_model = "gemini-3.6-flash"
    fallback_model = "gemini-3.5-flash"

    def __init__(self, model: str = default_model):
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for the gemini backend")
        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def find_regions(
        self, image_bytes: bytes, mime_type: str, *, cleanup: bool = False
    ) -> list[TextRegion]:
        prompt = CLEANUP_REGION_PROMPT if cleanup else REGION_PROMPT
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.ServerError as error:
            if error.code != 503 or self.model == self.fallback_model:
                raise
            print(
                f"{self.model} is unavailable; retrying with {self.fallback_model}"
            )
            response = self.client.models.generate_content(
                model=self.fallback_model, contents=contents, config=config
            )
        return parse_regions(response.text)
