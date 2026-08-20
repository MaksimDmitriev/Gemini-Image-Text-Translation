import unittest
from types import SimpleNamespace

from PIL import Image, ImageDraw

from drawing_translator.models import TextRegion
from drawing_translator.providers import BACKENDS
from drawing_translator.providers.gemini import parse_regions
from drawing_translator.providers.google_openai import GoogleOpenAIProvider
from drawing_translator.rendering import (
    _cell_box,
    _line_masks,
    normalized_box,
    normalized_rotation,
)


class CoreTests(unittest.TestCase):
    def test_backends_are_exposed(self):
        self.assertEqual(BACKENDS, ("gemini", "google-openai"))

    def test_parse_gemini_regions(self):
        result = parse_regions(
            '```json\n[{"source":"ТУ","translation":"TU",'
            '"box_2d":[10,20,30,40],"target_box_2d":[5,15,35,45],'
            '"rotation_degrees":90}]\n```'
        )
        self.assertEqual(
            result,
            [TextRegion("ТУ", "TU", (10, 20, 30, 40), (5, 15, 35, 45), 90)],
        )

    def test_normalized_box_converts_to_pixels(self):
        self.assertEqual(
            normalized_box((100, 200, 500, 800), 1000, 500),
            (200, 50, 800, 250),
        )

    def test_rotation_is_rounded_to_quarter_turn(self):
        self.assertEqual(normalized_rotation(88), 90)
        self.assertEqual(normalized_rotation("bad"), 0)

    def test_cell_expansion_stays_in_text_band(self):
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 190, 90), outline="black", width=2)
        expanded = _cell_box((80, 40, 120, 60), _line_masks(image), image.size)
        self.assertLess(expanded[0], 80)
        self.assertGreater(expanded[2], 120)
        self.assertGreater(expanded[1], 10)
        self.assertLess(expanded[3], 90)

    def test_openai_translation_results_are_restored_to_input_order(self):
        class FakeResponses:
            @staticmethod
            def create(**kwargs):
                self.assertEqual(kwargs["model"], "gpt-5.6-terra")
                return SimpleNamespace(
                    output_text=(
                        '{"translations":['
                        '{"id":1,"translation":"Second"},'
                        '{"id":0,"translation":"First"}]}'
                    )
                )

        provider = GoogleOpenAIProvider.__new__(GoogleOpenAIProvider)
        provider.model = "gpt-5.6-terra"
        provider.openai_client = SimpleNamespace(responses=FakeResponses())
        self.assertEqual(provider._translate(["Первый", "Второй"]), ["First", "Second"])


if __name__ == "__main__":
    unittest.main()
