# Video Analyzer Streamlit App

Streamlit application that:
- Accepts an uploaded video file
- Extracts format, audio, and video metadata via `ffmpeg`
- Samples frames to generate a natural-language caption (BLIP image captioning)
- Scores the caption against the video using CLIP similarity
- Estimates a motion score from frame differences

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### System Dependencies

FFmpeg is required for frame extraction.  
On macOS (Homebrew):

```bash
brew install ffmpeg
```

Streamlit Community Cloud installs `ffmpeg` automatically via `packages.txt`.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repo, pointing to `app.py`.
3. The service installs Python dependencies from `requirements.txt` and system packages from `packages.txt` automatically.

## Environment Notes

The Hugging Face models download at runtime and run on CPU. No secrets or tokens are required. Ensure your video files are reasonably sized to keep processing responsive.

