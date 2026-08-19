# Gemini Image Text Translation

A small Python script that finds Russian text in a technical drawing, translates it
to English with Gemini, removes the Russian text, and fits the translation into the
same table cell or label area. It uses one Gemini API call per image.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`, then run:

```bash
python translate_drawing.py drawing.png output/drawing_en.png
```

The default model is `gemini-2.5-flash`. It can be changed with `--model`.

## Notes

- Gemini returns normalized bounding boxes for the original text and its safe writing
  area. The script converts them to pixels, estimates the local paper colour, erases
  the source, and chooses the largest font that fits.
- Standalone numbers and existing Latin text are intentionally left unchanged.
- Very dense or low-resolution drawings may need a second pass or prompt tuning.
