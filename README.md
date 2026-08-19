# Gemini Image Text Translation

A small Python script that finds Russian text in a technical drawing, translates it
to English with Gemini, removes the Russian text, and fits the translation into the
same table cell or label area. It uses one Gemini API call per image.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export your Gemini API key. For example, add this to `~/.zshrc`:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Load it into the current terminal:

```bash
source ~/.zshrc
```

Put the drawings at `input/1.jpg`, `input/2.jpg`, and `input/3.jpg`, then run:

```bash
python translate_drawing.py input/1.jpg output/1_en.png
python translate_drawing.py input/2.jpg output/2_en.png
python translate_drawing.py input/3.jpg output/3_en.png
```

The default model is `gemini-3.6-flash`. If it returns a temporary `503` capacity
error, the script automatically retries with `gemini-3.5-flash-lite`. The primary
model can be changed with `--model`.

## Notes

- The output is PNG because translated text and thin drawing lines benefit from
  lossless compression. Saving back to JPEG would compress the image again,
  potentially introducing blur and artifacts around text and lines. In testing,
  JPEG output looked visibly worse and the translated text also overflowed some
  cells; the overflow may come from variation in Gemini's detected bounding boxes
  rather than from JPEG compression itself.
- Gemini returns normalized bounding boxes for the original text and its safe writing
  area. The script converts them to pixels, estimates the local paper colour, erases
  the source, and chooses the largest font that fits. Horizontal and 90-degree rotated
  labels keep their original orientation.
- Standalone numbers and existing Latin text are intentionally left unchanged.
- Very dense or low-resolution drawings may need a second pass or prompt tuning.
