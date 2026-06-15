# IndexTTS2 Metal

[中文](README.zh.md) | English

IndexTTS2 Metal is a native Apple Silicon runtime for IndexTTS2. It starts from
the PyTorch IndexTTS2 model layout and converts the model into a fixed MIT2
bundle that can be loaded by a C++/Objective-C++ runtime using Metal and
MetalPerformanceShaders.

## Features

- Apple Silicon first: native Metal kernels and MPS operators for macOS on M
  series chips.
- PyTorch-to-native workflow: convert IndexTTS2 checkpoints into an mmap-friendly
  MIT2 model bundle with aligned tensor payloads and manifest validation.
- Native TTS runtime: synthesize speech from a model bundle, a voice bundle, and
  input text without running the original PyTorch inference path.
- Native voice clone path: build a voice bundle from reference audio when the
  clone encoder weights are present in the model bundle.
- HTTP server and embedded web UI: run the runtime as a local service, optionally
  exposing the browser-based management UI.
- Runtime quality/speed control: adjust CFM synthesis steps for the tradeoff
  between speed and acoustic quality.

## Performance

This project rewrites the IndexTTS2 inference engine in C++ for Apple Silicon.
In local tests, the native Metal runtime reduced RTF compared with the PyTorch
IndexTTS2 runtime:

| Hardware | PyTorch IndexTTS2 RTF | IndexTTS2 Metal RTF |
| --- | ---: | ---: |
| M1 Max 64G | 2.28 | 1.71 |
| M3 Ultra 256G | 2.20 | 0.82 |

## Requirements

- Apple Silicon Mac.
- macOS with Metal and MetalPerformanceShaders support.
- Xcode Command Line Tools.
- CMake 3.20 or newer.
- C++17 / Objective-C++17 compiler from Apple Clang.
- Python 3.10 or newer for conversion and tooling.
- Python packages from `pyproject.toml`; PyTorch-related extras are only needed
  when reading original PyTorch checkpoints or `.pt` voice profiles.
- A converted MIT2 model bundle, usually stored at `bin/`.
- One or more converted voice bundles, either imported into the server voice
  store or passed directly to `--tts`.

Install the Python package locally when you need the conversion tools:

```bash
python -m pip install -e ".[torch,dev]"
```

## Model Resources

`mtts` requires the converted MIT2 model files at runtime. Use one of the
following model sources:

1. Hugging Face: [raoqu/index-tts2-metal](https://huggingface.co/raoqu/index-tts2-metal)
2. ModelScope: [iwannaido/index-tts2-metal](https://modelscope.cn/models/iwannaido/index-tts2-metal)

Download the model files into the default `bin/` directory:

```bash
mkdir -p bin
git clone https://huggingface.co/raoqu/index-tts2-metal bin
```

Or download to any directory and pass it explicitly with `--model_bundle` in
server/web mode:

```bash
./build/mtts --web \
  --host 127.0.0.1 \
  --port 3456 \
  --model_bundle /path/to/index-tts2-metal
```

For direct CLI synthesis, pass the model bundle path as the first positional
argument after `--tts`:

```bash
./build/mtts --tts /path/to/index-tts2-metal /path/to/voice-bundle "Text to synthesize." out.wav
```

## Build

Build the native runtime:

```bash
./build.sh
```

The script configures CMake in `build/` and builds the `mtts` executable:

```text
build/mtts
```

You can override the build directory, build type, or parallelism:

```bash
BUILD_DIR=build-release CMAKE_BUILD_TYPE=Release JOBS=8 ./build.sh
```

## Run

### Convert a model bundle

If you already have a converted MIT2 model bundle, skip this step. Otherwise,
convert the PyTorch IndexTTS2 checkpoint directory into a native bundle:

```bash
python -m metal_indextts2.tools.convert_model \
  --checkpoint-dir /path/to/index-tts2/checkpoints \
  --output bin \
  --force-dtype f16
```

Additional clone-time components such as semantic codec, BigVGAN, CAMPPlus, and
W2V-BERT can be included through the converter options when needed.

### Start the HTTP API

```bash
./build/mtts --server \
  --host 127.0.0.1 \
  --port 3456 \
  --model_bundle bin
```

### Start the web UI

`--web` implies `--server` and exposes the embedded UI at `/web`:

```bash
./build/mtts --web \
  --host 127.0.0.1 \
  --port 3456 \
  --model_bundle bin
```

Open:

```text
http://127.0.0.1:3456/web
```

### Run TTS from the CLI

```bash
./build/mtts --tts bin voices/bundles/qin "今天的天气不错，我们去划船吧。" out.wav
```

Lower CFM step counts are faster; higher values usually preserve more acoustic
quality. The runtime option is spelled `--cfm_steps`:

```bash
./build/mtts --cfm_steps 16 --tts bin voices/bundles/qin "今天的天气不错，我们去划船吧。" out.wav
```

### Clone a voice

Use a reference WAV file to create a native voice bundle:

```bash
./build/mtts --clone bin sample/reference.wav voices/bundles/demo
```

### Runtime options

Only the main product-facing options are listed here.

| Option | Description |
| --- | --- |
| `--web` | Starts the HTTP server and enables the embedded web UI at `/web`. |
| `--server` | Starts the HTTP API without exposing the web UI. |
| `--host HOST` | Bind address for server/web mode. Default: `127.0.0.1`. |
| `--port PORT` | Bind port for server/web mode. Default: `3456`. |
| `--cfm_steps N` | Sets CFM synthesis steps. Valid range: `12` to `25`. |
| `--clone MODEL_BUNDLE AUDIO_WAV OUTPUT_VOICE_BUNDLE` | Creates a native voice bundle from a reference WAV file. |
| `--tts MODEL_BUNDLE VOICE_BUNDLE TEXT OUTPUT_WAV` | Synthesizes text to a WAV file with a model bundle and voice bundle. |
| `--model_bundle DIR` | Model bundle directory used by server/web mode. Default: `MODEL_BUNDLE` or `bin`. |

## Screenshots

### Voice Management

The Voices page lists imported and cloned voices, supports creating a voice from
reference audio, importing an existing bundle, and testing TTS output with the
selected voice.

![Voice management](snapshot/voices.jpg)

### Web Dashboard

The Status dashboard shows runtime health, queue state, the current job, recent
jobs, elapsed time, and RTF metrics for completed synthesis requests.

![Web dashboard](snapshot/status.jpg)
