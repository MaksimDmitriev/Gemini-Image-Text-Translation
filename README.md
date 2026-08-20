# Image Text Translation

A small Python script that finds Russian text in a technical drawing, translates it
to English, removes the Russian text, and fits the translation into the same table
cell or label area. Model-specific code is isolated behind a provider interface.

Two backends are available:

- `gemini` uses Gemini vision for detection, translation, and bounding boxes.
- `google-openai` uses Google Cloud Vision for OCR geometry, GPT-5.6 Terra for
  translation, and OpenCV to conservatively identify table-cell boundaries.

Both backends make an initial pass and one verification pass by default.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include both backends. You only need credentials for the backend
you select.

### Gemini credentials

Export your Gemini API key in the current terminal:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

To persist the key for future terminals, add the same export to `~/.zshrc`. If you
reload `~/.zshrc` in the current terminal, do it before activating the virtual
environment because shell configuration may reset `PATH`:

```bash
deactivate 2>/dev/null || true
source ~/.zshrc
source .venv/bin/activate
```

### Google OCR + OpenAI credentials

For local development, you do not need to download a credential file. The easiest
route is signing in with the Google Cloud CLI.

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and create or
   select a project.
2. Ensure billing is enabled.
3. Enable the [Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com).
4. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) if
   `gcloud` is not available.

Initialize the CLI, then configure local Application Default Credentials. The commands
below use this project's Google Cloud project ID:

```bash
# Initialize gcloud and sign in
gcloud init

# 1. Select the correct project
gcloud config set project gemini-image-text-translation

# 2. Enable Cloud Vision OCR
gcloud services enable vision.googleapis.com \
  --project=gemini-image-text-translation

# 3. Ensure an old service-account path does not override local credentials
unset GOOGLE_APPLICATION_CREDENTIALS

# 4. Create local Application Default Credentials
gcloud auth application-default login

# 5. Assign the project used for billing and quota
gcloud auth application-default set-quota-project \
  gemini-image-text-translation

# 6. Verify Google authentication
gcloud auth application-default print-access-token >/dev/null \
  && echo "Google credentials OK"

# 7. Verify the selected project
gcloud config get-value project

# 8. Verify the Vision API is enabled
gcloud services list --enabled \
  --project=gemini-image-text-translation \
  --filter="config.name:vision.googleapis.com"
```

Then export an OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

Put drawings at `input/1.jpg`, `input/2.jpg`, and `input/3.jpg`, then process the
whole folder sequentially with the backward-compatible Gemini default:

```bash
python translate_drawing.py input output
```

Select Google OCR plus OpenAI for either a folder or one image:

```bash
python translate_drawing.py input output --backend google-openai
python translate_drawing.py input/1.jpg output/1_en.png --backend google-openai
```

The results are saved as `output/1_en.png`, `output/2_en.png`, and
`output/3_en.png`. PNG, JPEG, and WebP inputs are supported. Folder scanning is
non-recursive and files are processed one at a time. If the output is omitted, a
folder uses `output/`, while one image uses `<original-name>_translated.png`.

Override the model or number of verification passes when needed:

```bash
python translate_drawing.py input output \
  --backend google-openai \
  --model gpt-5.6-luna \
  --verification-passes 1
```

Backend defaults are `gemini-3.6-flash` and `gpt-5.6-terra`. For Gemini only, a
temporary `503` from the default model is retried with `gemini-3.5-flash`.

## Notes

- Output is PNG because translated text and thin drawing lines benefit from lossless
  compression. Saving back to JPEG would introduce another lossy compression pass.
- Providers return a shared normalized-region type. The renderer converts regions to
  pixels, erases the source, and fits the translation without exceeding the source
  text's approximate font size. Quarter-turn labels retain their orientation.
- Google OCR provides tight text polygons. OpenCV expands a writing area only when it
  can find all four surrounding table lines; otherwise the tight OCR box is retained.
- Standalone numbers and existing Latin text are left unchanged.
- Very dense or low-resolution drawings may still need OCR or prompt tuning.
