# RTlation

*Real-time speech translation in your own voice – runs locally on your GPU.*

**Language:** English · [Deutsch](README.de.md)

**Real Time / Translation** – speak into your microphone in one language and the translation comes out in another, spoken in your own voice if you like. Everything runs locally on your own computer; nothing is sent to the cloud.

The user interface is currently in German. The labels in bold below are the ones you will see on screen, with a translation in brackets.

## Setup (Windows, NVIDIA GPU)

Requirements: an NVIDIA graphics card with at least 12 GB of memory and [Ollama](https://ollama.com) installed and running.

```bat
setup.bat
```

The script puts everything inside the project folder: `uv` and Python 3.11 in `.tools`, the packages in `.venv`, the models in `models`. Only the translation model is stored by Ollama. You can run the script again at any time; it skips whatever is already there.

| Option | Effect |
|---|---|
| `-LimitRate 7M` | limits the large downloads to 7 MB/s |
| `-Stt large-v3-turbo,large-v3` | speech recognition models (default: only `large-v3-turbo`); `large-v3` is bigger (3 GB) |
| `-Tts chatterbox,qwen3` | installs both voice models (default: only `chatterbox`) |
| `-Tts none` | no spoken output |

## Running

```bat
run.bat
```

Your browser opens `http://127.0.0.1:8765`. Press **Start** and speak.

## Discord, TeamSpeak and similar

For others to hear the translation, Windows needs a virtual microphone. Install [VB-Cable](https://vb-audio.com/Cable/) once (free, it is a driver) and restart your computer. RTlation then picks "CABLE Input" as its output device by itself; in Discord or TeamSpeak, choose "CABLE Output" as your microphone. Tick **Mithören** (listen along) to also hear the translation on your normal speakers or headphones.

## Your own voice

Next to the **Stimme** (voice) field, the plus button opens a dialog: read the text aloud (10 to 15 seconds) or pick an existing recording, listen to it, save it. The voice then appears in the list. It is stored as a WAV file in your user profile (Windows: `%APPDATA%\RTlation\voices`), not in the project folder, so it does not travel along when you share the folder; you can change the location with `voices_dir` in `config.yaml`. Silence at the start and end is trimmed, and a transcript is created automatically.

A German recording also works for other target languages. Chatterbox adds a watermark to its output (inaudible, according to the maker) that marks it as AI-generated.

**Ausdruck** (expression) sets how lively the voice sounds. With **Auto** ticked, each sentence is additionally adjusted to how energetically you spoke it (loudness, pitch movement, pace).

## Voice models

| Choice | Behaviour |
|---|---|
| Chatterbox · schnell (fast) | Audio starts while the sentence is still being generated. Default. |
| Chatterbox · ganzer Satz (whole sentence) | The whole sentence is generated first, then played (the library's own method). |
| Qwen3-TTS | For comparison; slower than real time on an RTX 4070 Ti SUPER. |

Switching between the two Chatterbox modes takes effect immediately; switching to Qwen3-TTS loads a different model.

## Settings

All settings are documented in [config.yaml](config.yaml): languages, models, port and the parameters of the pause detection. Languages, microphone, speaking pause, translation style, voice, voice model and colour theme can also be changed directly in the interface. The footer shows how much GPU memory the current selection needs.

## How it works

```
Microphone ─▶ Pause detection ─▶ Speech recognition ─▶ Translation ─▶ Speech output ─▶ Speakers
(browser)     Silero VAD         faster-whisper        TranslateGemma   Chatterbox       (browser)
                                 (large-v3-turbo)      (via Ollama)     or Qwen3-TTS
```

| Path | Contents |
|---|---|
| `src/rtlation/engine.py` | the pipeline: audio in, events out |
| `src/rtlation/stages/` | the individual stages (`vad`, `stt`, `translate`, `tts`) |
| `src/rtlation/server.py` | web server and WebSocket interface |
| `src/rtlation/web/` | user interface |
| `workers/` | voice models as separate processes (`base.py` describes the protocol) |
| `voices/` | the bundled default voice |
| `tests/` | tests and a German test recording |

The interface always talks to the server over a WebSocket, so the same code runs locally and on a remote machine.

Each voice model runs in its own process with its own Python environment (`.tools/venvs/<name>`), because the models' dependencies conflict with each other. Which model speaks is set in `config.yaml` under `tts.engine` and can be switched in the interface.

## Testing

```bat
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe tests\stream_sample.py --target es --save tests\out
.venv\Scripts\python.exe tests\bench_tts.py chatterbox --check
```

The second command plays the test recording against the running server and shows text, translation and delay; with `--save` the spoken output is written as WAV files. The third measures a voice model on its own (the server must not be running) and uses Whisper to check that the speech matches the text.

## Roadmap

1. ✅ Microphone → text → translation, with user interface
2. ✅ Spoken translation
3. ✅ Your own voice (voice cloning) and streamed output
4. On hold: server mode with Docker (also needs about 10 GB of GPU memory)
5. ✅ Virtual microphone for Discord, TeamSpeak etc. via VB-Cable

## License

RTlation's code is released under the MIT license (see [LICENSE](LICENSE)). The models downloaded by `setup.bat` come with their own terms:

| Model | License |
|---|---|
| Whisper large-v3 / large-v3-turbo (OpenAI) | MIT |
| Silero VAD | MIT |
| TranslateGemma (Google, via Ollama) | Gemma Terms of Use |
| Chatterbox (Resemble AI) | MIT; output carries a watermark |
| Qwen3-TTS (Alibaba) | Apache 2.0 |
