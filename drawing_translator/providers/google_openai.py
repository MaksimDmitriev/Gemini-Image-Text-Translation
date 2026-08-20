"""Google Cloud Vision OCR plus OpenAI translation provider."""

import json
import math
import os
import re
from collections.abc import Iterable

from drawing_translator.models import Box, TextRegion
from drawing_translator.prompts import TRANSLATION_PROMPT


CYRILLIC = re.compile(r"[\u0400-\u04ff]")


def _normalized_box(vertices: Iterable[object], width: int, height: int) -> Box:
    points = [(getattr(v, "x", 0), getattr(v, "y", 0)) for v in vertices]
    if not points:
        raise ValueError("Google OCR returned a region without vertices")
    xs, ys = zip(*points)
    return (
        round(min(ys) * 1000 / height),
        round(min(xs) * 1000 / width),
        round(max(ys) * 1000 / height),
        round(max(xs) * 1000 / width),
    )


def _rotation(vertices: list[object]) -> int:
    if len(vertices) < 2:
        return 0
    dx = getattr(vertices[1], "x", 0) - getattr(vertices[0], "x", 0)
    dy = getattr(vertices[1], "y", 0) - getattr(vertices[0], "y", 0)
    return (round(math.degrees(math.atan2(dy, dx)) / 90) * 90) % 360


class GoogleOpenAIProvider:
    """Use deterministic OCR geometry and an LLM only for translation."""

    name = "google-openai"
    default_model = "gpt-5.6-terra"

    def __init__(self, model: str = default_model):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for google-openai")
        try:
            from google.auth.exceptions import DefaultCredentialsError
            from google.cloud import vision
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "google-openai requires google-cloud-vision and openai; "
                "run pip install -r requirements.txt"
            ) from error

        self.model = model
        self.vision = vision
        try:
            self.ocr_client = vision.ImageAnnotatorClient()
        except DefaultCredentialsError as error:
            raise RuntimeError(
                "Google Cloud credentials are required for google-openai; run "
                "'gcloud auth application-default login' or set "
                "GOOGLE_APPLICATION_CREDENTIALS"
            ) from error
        self.openai_client = OpenAI()

    def _detect(self, image_bytes: bytes) -> list[tuple[str, Box, int]]:
        response = self.ocr_client.document_text_detection(
            image=self.vision.Image(content=image_bytes)
        )
        if response.error.message:
            raise RuntimeError(f"Google Cloud Vision OCR failed: {response.error.message}")

        regions: list[tuple[str, Box, int]] = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    words = [
                        "".join(symbol.text for symbol in word.symbols)
                        for word in paragraph.words
                    ]
                    source = " ".join(word for word in words if word).strip()
                    if not CYRILLIC.search(source):
                        continue
                    vertices = [
                        vertex
                        for word in paragraph.words
                        for vertex in word.bounding_box.vertices
                    ]
                    first_vertices = (
                        list(paragraph.words[0].bounding_box.vertices)
                        if paragraph.words
                        else []
                    )
                    box = _normalized_box(vertices, page.width, page.height)
                    regions.append((source, box, _rotation(first_vertices)))
        return regions

    def _translate(self, sources: list[str]) -> list[str]:
        if not sources:
            return []
        payload = [{"id": index, "source": source} for index, source in enumerate(sources)]
        schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "translation": {"type": "string"},
                        },
                        "required": ["id", "translation"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        }
        response = self.openai_client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=TRANSLATION_PROMPT,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "drawing_translations",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        parsed = json.loads(response.output_text)
        by_id = {
            item["id"]: item["translation"] for item in parsed["translations"]
        }
        missing = [index for index in range(len(sources)) if index not in by_id]
        if missing:
            raise ValueError(f"OpenAI omitted translation ids: {missing}")
        return [str(by_id[index]) for index in range(len(sources))]

    def find_regions(
        self, image_bytes: bytes, mime_type: str, *, cleanup: bool = False
    ) -> list[TextRegion]:
        del mime_type, cleanup
        detected = self._detect(image_bytes)
        translations = self._translate([item[0] for item in detected])
        return [
            TextRegion(
                source=source,
                translation=translation,
                box_2d=box,
                target_box_2d=box,
                rotation_degrees=rotation,
            )
            for (source, box, rotation), translation in zip(detected, translations)
        ]
