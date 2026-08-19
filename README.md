# Gemini Image Text Translation

A small Python script that finds Russian text in a technical drawing, translates it
to English with Gemini, removes the Russian text, and fits the translation into the
same table cell or label area. It uses a second Gemini pass to catch small Cyrillic
text missed on the first pass and a third pass for a final check. A successful full
run normally uses three API calls per image. The output is saved after every pass; if
a cleanup pass receives a temporary `503`, the latest usable image is kept.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The only runtime dependencies are the official `google-genai` SDK and Pillow.

Export your Gemini API key. For example, add this to `~/.zshrc`:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Load it into the current terminal:

```bash
source ~/.zshrc
```

Put the drawings at `input/1.jpg`, `input/2.jpg`, and `input/3.jpg`, then process the
whole folder sequentially:

```bash
python translate_drawing.py input output
```

The results are saved as `output/1_en.png`, `output/2_en.png`, and
`output/3_en.png`. PNG, JPEG, and WebP inputs are supported. Folder scanning is
non-recursive and files are processed one at a time. A single image can still be
processed with `python translate_drawing.py input/1.jpg output/1_en.png`.

The default model is `gemini-3.6-flash`. If it returns a temporary `503` capacity
error, the script automatically retries with `gemini-3.5-flash`. The primary model
can be changed with `--model`. A fallback adds one API call for the affected pass.
The normal three-pass run uses three calls per image; if every pass needs the
fallback, the maximum is six calls per image.

## Notes

- The output is PNG because translated text and thin drawing lines benefit from
  lossless compression. Saving back to JPEG would compress the image again,
  potentially introducing blur and artifacts around text and lines. In testing,
  JPEG output looked visibly worse and the translated text also overflowed some
  cells; the overflow may come from variation in Gemini's detected bounding boxes
  rather than from JPEG compression itself.
- Gemini returns normalized bounding boxes for the original text and its safe writing
  area. The script converts them to pixels, erases the source with white, and fits the
  translation without exceeding the source text's approximate font size. Labels
  rotated by 90-degree increments keep their original orientation.
- Standalone numbers and existing Latin text are intentionally left unchanged.
- Very dense or low-resolution drawings may still need prompt tuning.
