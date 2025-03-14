# Exscribe 📝

Exscribe **ex**tracts and tran**scribes** subtitles from videos using multimodal language models.

---
Why another subtitle extraction tool you might ask? Traditional OCR tools often struggle with video subtitles due to their size, font, and variable position -- resulting in frequent mis-spellings, missing words, and other errors. Exscribe aims to solve these issues by leveraging AI models for more accurate, coherent, and reliable results.


## Features

- Frame extraction and processing from videos
- AI-powered OCR (supports Gemini, OpenAI support planned)
- Subtitle merging and post-processing
- SRT file generation
- Multiple language support
- Debug mode for troubleshooting

## Installation

Exscribe is available on PyPI and can be installed via pip:

```bash
pip install exscribe
```

## Usage Guide

Exscribe is designed to be simple yet powerful. At its core, you can extract subtitles from any video file with a single command:

```bash
exscribe video_file.mp4
```

This will process your video using default settings (Gemini AI, English language) and create an SRT file with the same name as your input video.

### How It Works

1. **Frame Extraction** from video file
2. **Frame Processing** with optional edge detection
3. **Transcription** via AI provider
4. **Subtitle Generation** through merging frames with the same text
5. **SRT Creation** with proper formatting

### Configuration Options

Exscribe can be customized to handle different video types, languages, and processing needs:

**Language Selection:** Specify the language of the subtitles you're extracting with the `--language` option. This helps the AI model accurately recognize text.

**AI Provider:** Choose between AI providers with the `--provider` option. Currently, Gemini is fully supported with OpenAI integration planned.

**Performance Tuning:** Adjust how Exscribe processes videos with options like:

- `--batch_size` to control memory usage and processing speed
- `--frame_skip` to reduce processing time by analyzing fewer frames
- `--similarity_threshold` to fine-tune frame comparison sensitivity

**Troubleshooting:** Enable `--debug` mode to save intermediate files which can help diagnose issues or improve results.

### API Key Management

Exscribe requires API keys for the AI providers. You can provide them in two ways:

```bash
# Method 1: Command line arguments
exscribe video.mp4 --gemini_api_key YOUR_KEY

# Method 2: Environment variables (recommended for security)
export GEMINI_API_KEY=your_api_key
export OPENAI_API_KEY=your_api_key
```


## Comparison with Alternatives

| Feature | Exscribe | Traditional OCR | Subtitle Editors |
|---------|:------:|:--------------:|:----------------:|
| High accuracy with stylized fonts | ✅ | ❌ | ❌ |
| Context-aware text recognition | ✅ | ❌ | ✅ |
| Wide language support | ✅ ✅| ⚠️ | ✅ |
| Effective with burned-in subtitles | ✅  | ⚠️ | ❌ |
| Special character recognition | ✅ | ❌ | ✅ |
| Intelligent subtitle merging | ✅ | ❌ | ⚠️ |
| Fully automated processing | ✅ | ⚠️ | ❌ |
| Simple setup | ✅ | ❌ | ✅ |
| No API costs | ⚠️* | ✅ | ✅ |
| Works without internet | ❌ | ✅ | ✅ |
| Fast processing | ⚠️ | ✅ ✅ | ❌ |

✅ = Fully supported  |  ⚠️ = Partially supported  |  ❌ = Not supported

*Free quotas available for Gemini AI, OpenAI requires billing account.

### Practical Examples

**Basic Processing:**

```bash
# Process a single video with default settings
exscribe video.mp4
```

**Batch Processing:**

```bash
# Process all videos in a directory
exscribe /path/to/videos/
```

**Advanced Configuration:**

```bash
# Extract English subtitles with custom processing settings
exscribe documentary.mp4 --language English --batch_size 32 --frame_skip 15
```

**Testing New Providers:**

```bash
# Try OpenAI with debug mode for troubleshooting
exscribe video.mp4 --provider openai --debug
```

**Keeping only unique structurally similar frames:**

```bash
# For videos with hard-to-read subtitles
exscribe difficult_video.mp4 --edge_detection --similarity_threshold 0.1
```
