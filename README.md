# RTlation

*Real-time speech translation in your own voice – runs locally on your GPU.*

**Real Time / Translation** – spricht man in einer Sprache ins Mikrofon, kommt die Übersetzung in einer anderen heraus. Alles läuft lokal auf dem eigenen Rechner.

Stand: **Schritt 3** – Sprache wird erkannt, übersetzt und vorgelesen, auf Wunsch mit der eigenen Stimme.

## Einrichten (Windows, NVIDIA-Grafikkarte)

Voraussetzungen: NVIDIA-Grafikkarte mit mindestens 12 GB Speicher und ein installiertes, gestartetes [Ollama](https://ollama.com).

```bat
setup.bat
```

Das Skript legt alles im Projektordner ab: `uv` und Python 3.11 in `.tools`, die Pakete in `.venv`, die Modelle in `models`. Nur das Übersetzungsmodell liegt im Ollama-Speicher. Das Skript kann jederzeit erneut gestartet werden und überspringt, was schon da ist.

| Option | Wirkung |
|---|---|
| `-LimitRate 7M` | drosselt die grossen Downloads auf 7 MB/s |
| `-Stt large-v3-turbo,large-v3` | Modelle für die Spracherkennung (Standard: nur `large-v3-turbo`); `large-v3` ist grösser (3 GB) |
| `-Tts chatterbox,qwen3` | richtet beide Sprachmodelle ein (Standard: nur `chatterbox`) |
| `-Tts none` | ohne Sprachausgabe |

## Starten

```bat
run.bat
```

Der Browser öffnet `http://127.0.0.1:8765`. Dort **Start** drücken und sprechen.

## Discord, TeamSpeak und Co.

Damit andere die Übersetzung hören, braucht Windows ein virtuelles Mikrofon. Dafür einmal [VB-Cable](https://vb-audio.com/Cable/) installieren (kostenlos, ein Treiber) und den Rechner neu starten. RTlation wählt danach „CABLE Input“ von selbst als Ausgabegerät; in Discord oder TeamSpeak stellt man „CABLE Output“ als Mikrofon ein. Mit **Mithören** hört man die Übersetzung zusätzlich auf dem eigenen Standardgerät.

## Eigene Stimme

Neben dem Feld **Stimme** öffnet der Plus-Knopf einen Dialog: Text vorlesen (10 bis 15 Sekunden) oder eine vorhandene Aufnahme als Datei wählen, anhören, speichern. Die Aufnahme steht danach in der Auswahl. Sie liegt als WAV im Benutzerprofil (Windows: `%APPDATA%\RTlation\voices`) und nicht im Projektordner, damit sie beim Weitergeben des Ordners nicht mitwandert; der Ort lässt sich mit `voices_dir` in `config.yaml` ändern. Stille am Anfang und Ende wird abgeschnitten, eine Abschrift entsteht automatisch.

Eine deutsche Aufnahme funktioniert auch für andere Zielsprachen. Chatterbox versieht die Ausgabe mit einem Wasserzeichen (laut Hersteller unhörbar), das sie als künstlich erzeugt kennzeichnet.

## Sprachmodelle

| Auswahl | Verhalten |
|---|---|
| Chatterbox · schnell | Der Ton kommt stückweise, während der Satz noch entsteht. Standard. |
| Chatterbox · ganzer Satz | Der Satz wird erst fertig berechnet und dann abgespielt (Verfahren der Bibliothek). |
| Qwen3-TTS | Zum Vergleich; auf einer RTX 4070 Ti SUPER langsamer als gesprochen. |

Der Wechsel zwischen den beiden Chatterbox-Arten ist sofort wirksam, der Wechsel zu Qwen3-TTS lädt ein anderes Modell.

## Einstellungen

Alle Einstellungen stehen kommentiert in [config.yaml](config.yaml): Sprachen, Modelle, Port und die Parameter der Pausenerkennung. Sprachen, Mikrofon, Sprechpause, Stimme, Sprachmodell und Farbpalette lassen sich auch direkt in der Oberfläche umstellen.

## Aufbau

```
Mikrofon ─▶ Pausenerkennung ─▶ Spracherkennung ─▶ Übersetzung ─▶ Sprachausgabe ─▶ Lautsprecher
(Browser)    Silero VAD        faster-whisper     TranslateGemma   Chatterbox        (Browser)
                               (large-v3-turbo)   (über Ollama)    oder Qwen3-TTS
```

| Pfad | Inhalt |
|---|---|
| `src/rtlation/engine.py` | Pipeline: Audio rein, Ereignisse raus |
| `src/rtlation/stages/` | die einzelnen Stufen (`vad`, `stt`, `translate`, `tts`) |
| `src/rtlation/server.py` | Web-Server und WebSocket-Schnittstelle |
| `src/rtlation/web/` | Oberfläche |
| `workers/` | Sprachmodelle als eigene Prozesse (`base.py` beschreibt das Protokoll) |
| `voices/` | Referenzaufnahmen für Stimmen |
| `tests/` | Tests und eine deutsche Test-Aufnahme |

Die Oberfläche spricht immer über WebSocket mit dem Server. Lokal und später auf einem entfernten Server läuft damit derselbe Code.

Jedes Sprachmodell läuft in einem eigenen Prozess mit eigener Python-Umgebung (`.tools/venvs/<name>`), weil sich die Abhängigkeiten der Modelle gegenseitig ausschliessen. Welches Modell spricht, steht in `config.yaml` unter `tts.engine` und lässt sich in der Oberfläche umschalten.

## Testen

```bat
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe tests\stream_sample.py --target es --save tests\out
.venv\Scripts\python.exe tests\bench_tts.py chatterbox --check
```

Der zweite Befehl spielt die Test-Aufnahme gegen den laufenden Server und zeigt Text, Übersetzung und Verzögerung; mit `--save` landet die Sprachausgabe als WAV im Ordner. Der dritte misst ein Sprachmodell für sich (der Server darf dabei nicht laufen) und prüft mit Whisper, ob das Gesprochene zum Text passt.

## Fahrplan

1. ✅ Mikrofon → Text → Übersetzung, mit Oberfläche
2. ✅ Sprachausgabe der Übersetzung
3. ✅ Eigene Stimme (Stimmklon) und stückweise Ausgabe
4. Zurückgestellt: Server-Betrieb mit Docker (braucht ebenfalls rund 10 GB Grafikspeicher)
5. ✅ Virtuelles Mikrofon für Discord, TeamSpeak usw. über VB-Cable

## Lizenz

Der Code von RTlation steht unter der MIT-Lizenz (siehe [LICENSE](LICENSE)). Die Modelle, die `setup.bat` herunterlädt, haben eigene Bedingungen:

| Modell | Lizenz |
|---|---|
| Whisper large-v3 / large-v3-turbo (OpenAI) | MIT |
| Silero VAD | MIT |
| TranslateGemma (Google, über Ollama) | Gemma-Nutzungsbedingungen |
| Chatterbox (Resemble AI) | MIT; die Ausgabe trägt ein Wasserzeichen |
| Qwen3-TTS (Alibaba) | Apache 2.0 |
