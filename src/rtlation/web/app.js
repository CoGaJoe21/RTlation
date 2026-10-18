// RTlation - Oberfläche: nimmt das Mikrofon auf, schickt das Audio an den
// Server, zeigt Original und Übersetzung an und spielt die Sprachausgabe ab.

const SAMPLE_RATE = 16000;
const BLOCK = 1024;

const THEMES = [
  { id: "gold", label: "Schwarz & Gold", bg: "#0a0a0b", accent: "#d4af37" },
  { id: "midnight", label: "Mitternacht", bg: "#060a13", accent: "#5cc8ff" },
  { id: "emerald", label: "Smaragd", bg: "#060e0a", accent: "#3ddc97" },
  { id: "ruby", label: "Rubin", bg: "#0f0708", accent: "#ff5c7a" },
  { id: "ivory", label: "Elfenbein", bg: "#f5f0e6", accent: "#a67c00" },
];

const $ = (id) => document.getElementById(id);
const el = {
  toggle: $("toggle"), toggleLabel: $("toggle-label"),
  device: $("device"), source: $("source"), target: $("target"), swap: $("swap"),
  pause: $("pause"), pauseOut: $("pause-out"), file: $("file"), style: $("style"), sttModel: $("stt-model"),
  speak: $("speak"), output: $("output"), monitor: $("monitor"), engine: $("engine"),
  volume: $("volume"), volumeOut: $("volume-out"),
  expression: $("expression"), expressionOut: $("expression-out"), expressionAuto: $("expression-auto"),
  voice: $("voice"), voiceNew: $("voice-new"), voiceDelete: $("voice-delete"),
  dialog: $("voice-dialog"), voiceName: $("voice-name"), voiceRec: $("voice-rec"),
  voiceRecLabel: $("voice-rec-label"), voiceTime: $("voice-time"), voiceLevel: $("voice-level"),
  voiceFile: $("voice-file"), voicePreview: $("voice-preview"), voiceStatus: $("voice-status"),
  voiceCancel: $("voice-cancel"), voiceSave: $("voice-save"),
  level: $("level"), vad: $("vad"),
  rows: $("rows"), empty: $("empty"),
  headSource: $("head-source"), headTarget: $("head-target"),
  statCount: $("stat-count"), statLast: $("stat-last"), statAvg: $("stat-avg"),
  statGpu: $("stat-gpu"), statGpuBox: $("stat-gpu-box"),
  clear: $("clear"), toast: $("toast"), themes: $("themes"),
  stConn: $("st-conn"), stStt: $("st-stt"), stMt: $("st-mt"), stTts: $("st-tts"),
};

// Einstellungen überleben einen Neustart, wenn der Browser es zulässt.
const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(`rtlation.${key}`);
      return value === null ? fallback : value;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try { localStorage.setItem(`rtlation.${key}`, value); } catch { /* privates Fenster */ }
  },
};

let ws = null;
let connection = 0;       // zählt Verbindungen, weil der Server die Satz-IDs je Verbindung neu vergibt
let capture = null;       // laufende Aufnahme: { stop() }
let sttReady = false;
let ttsAvailable = false; // Sprachausgabe ist auf dem Server eingeschaltet
let languages = [];
const rows = new Map();
const delays = new Map(); // Satz -> Verzögerung in ms (bis zur Sprache, sonst bis zum Text)
let toastTimer = 0;

// ----------------------------------------------------------------- Paletten

function buildThemes() {
  const current = document.documentElement.dataset.theme;
  for (const theme of THEMES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "swatch";
    button.title = theme.label;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-label", theme.label);
    button.setAttribute("aria-checked", String(theme.id === current));
    button.style.setProperty("--sw-bg", theme.bg);
    button.style.setProperty("--sw-accent", theme.accent);
    button.addEventListener("click", () => {
      document.documentElement.dataset.theme = theme.id;
      store.set("theme", theme.id);
      for (const other of el.themes.children) other.setAttribute("aria-checked", String(other === button));
    });
    el.themes.append(button);
  }
}

// ------------------------------------------------------------------- Status

function setPill(pill, state, detail) {
  pill.dataset.state = state;
  pill.title = detail || "";
}

function applyStatus(status, engine, sttModel) {
  if (sttModel) el.sttModel.value = sttModel;
  el.sttModel.disabled = status.stt.state === "loading";
  setPill(el.stStt, status.stt.state, status.stt.detail);
  setPill(el.stMt, status.translate.state, status.translate.detail);
  setPill(el.stTts, status.tts.state, status.tts.detail);
  if (engine) el.engine.value = engine;
  sttReady = status.stt.state === "ready";
  refreshGpu();
  refreshToggle();
}

// Grafikspeicher: was die gewählten Modelle brauchen und was gerade belegt ist.
const formatGb = (mb) => (mb / 1024).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
let gpuTimer = 0;

async function refreshGpu() {
  clearTimeout(gpuTimer);
  gpuTimer = setTimeout(refreshGpu, 10000);
  let data;
  try {
    data = await (await fetch("/api/gpu")).json();
  } catch {
    return;
  }
  const { needed, usage } = data;
  const amount = `${needed.complete ? "ca. " : "über "}${formatGb(needed.total_mb)} GB`;
  const detail = document.createElement("small");
  detail.textContent = usage ? `belegt ${formatGb(usage.used_mb)} von ${formatGb(usage.total_mb)} GB` : "";
  el.statGpu.replaceChildren(amount, detail);
  // Knapp, sobald weniger als 1 GB frei ist.
  el.statGpu.classList.toggle("is-tight", Boolean(usage) && usage.total_mb - usage.used_mb < 1024);
  el.statGpuBox.title = [
    "Bedarf der gewählten Modelle (gemessen auf einer RTX 4070 Ti SUPER):",
    ...needed.parts.map((part) => `${part.label}: ${part.mb === null ? "nicht vermessen" : `${formatGb(part.mb)} GB`}`),
    "",
    "Belegt zählt alles auf der Grafikkarte mit, auch andere Programme.",
  ].join("\n");
  el.statGpuBox.hidden = false;
}

function refreshToggle() {
  const connected = ws && ws.readyState === WebSocket.OPEN;
  el.toggle.disabled = !capture && !(connected && sttReady);
}

function toast(message) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 7000);
}

// --------------------------------------------------------------- Verbindung

function connect() {
  setPill(el.stConn, "loading", "Verbindung wird aufgebaut");
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    connection++;
    setPill(el.stConn, "ready", "Mit dem Server verbunden");
    sendConfig();
    refreshToggle();
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") handle(JSON.parse(event.data));
    else playAudio(event.data);
  };
  ws.onclose = () => {
    setPill(el.stConn, "error", "Verbindung getrennt – neuer Versuch läuft");
    stopCapture(false);
    player.header = null;
    for (const row of el.rows.querySelectorAll(".is-live, .is-working")) row.classList.remove("is-live", "is-working");
    refreshToggle();
    setTimeout(connect, 1500);
  };
}

function sendJson(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function sendConfig() {
  sendJson({
    type: "config",
    source: el.source.value,
    target: el.target.value,
    min_silence_ms: Number(el.pause.value),
    speak: el.speak.checked,
    voice: el.voice.value,
    style: el.style.value,
    expression: Number(el.expression.value) / 100,
    auto_expression: el.expressionAuto.checked,
  });
}

// --------------------------------------------------------------- Wiedergabe

const canPickOutput = "setSinkId" in AudioContext.prototype;
const player = {
  context: null,
  gain: null,
  nextTime: 0,          // bis hierhin ist die Wiedergabe schon verplant
  header: null,         // Kopf des Audio-Blocks, der als Nächstes binär ankommt
  sources: new Set(),
};

// Zweite Wiedergabe auf dem Standardgerät, damit man mithört, was ins
// virtuelle Kabel (Discord, TeamSpeak) geht.
const monitor = { context: null, gain: null, nextTime: 0, sources: new Set() };
const CABLE = /CABLE Input/i;
const SETUP_CABLE = "__setup_cable__";
const CABLE_PAGE = "https://vb-audio.com/Cable/";

function ensureMonitor() {
  const wanted = el.monitor.checked && el.speak.checked && el.output.value !== "";
  if (!wanted) {
    if (monitor.context) monitor.context.close();
    monitor.sources.clear();
    Object.assign(monitor, { context: null, gain: null, nextTime: 0 });
    return;
  }
  if (!monitor.context) {
    monitor.context = new AudioContext({ latencyHint: "interactive" });
    monitor.gain = monitor.context.createGain();
    monitor.gain.connect(monitor.context.destination);
    applyVolume();
  }
  if (monitor.context.state === "suspended") monitor.context.resume();
}

function schedule(target, audio) {
  const source = target.context.createBufferSource();
  source.buffer = audio;
  source.connect(target.gain);
  const now = target.context.currentTime;
  const startAt = Math.max(now + 0.03, target.nextTime);
  source.start(startAt);
  target.nextTime = startAt + audio.duration;
  target.sources.add(source);
  source.onended = () => target.sources.delete(source);
  return { now, startAt };
}

// Browser erlauben Ton erst nach einem Klick; deshalb beim Start aufrufen.
function ensurePlayer() {
  ensureMonitor();
  if (!player.context) {
    player.context = new AudioContext({ latencyHint: "interactive" });
    player.gain = player.context.createGain();
    player.gain.connect(player.context.destination);
    applyVolume();
    applyOutput();
  }
  if (player.context.state === "suspended") player.context.resume();
}

function applyVolume() {
  el.volumeOut.textContent = `${el.volume.value} %`;
  if (player.gain) player.gain.gain.value = Number(el.volume.value) / 100;
  if (monitor.gain) monitor.gain.gain.value = Number(el.volume.value) / 100;
}

function applyOutput() {
  if (!player.context || !canPickOutput) return;
  player.context.setSinkId(el.output.value).catch(() => {
    toast("Das Ausgabegerät konnte nicht gewählt werden.");
  });
}

function stopPlayback() {
  for (const target of [player, monitor]) {
    for (const source of target.sources) source.stop();
    target.sources.clear();
    target.nextTime = 0;
  }
  for (const row of el.rows.querySelectorAll(".is-speaking")) row.classList.remove("is-speaking");
}

// Spielt einen Audio-Block lückenlos hinter dem vorherigen ab.
function playAudio(buffer) {
  const header = player.header;
  player.header = null;
  if (!header || !player.context || !el.speak.checked) return;

  const pcm = new Int16Array(buffer);
  const audio = player.context.createBuffer(1, pcm.length, header.sample_rate);
  const channel = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

  const { now, startAt } = schedule(player, audio);
  if (monitor.context) schedule(monitor, audio);

  const entry = rows.get(header.key);
  if (!entry) return;
  const waitMs = (startAt - now) * 1000;
  if (header.latency_ms !== undefined) {
    // Wartezeit dazuzählen, falls der vorherige Satz noch läuft.
    entry.heardMs = header.latency_ms + waitMs;
    delays.set(entry.key, entry.heardMs);
    renderMeta(entry);
    updateStats();
  }
  setTimeout(() => entry.node.classList.add("is-speaking"), waitMs);
  clearTimeout(entry.speakEnd);
  entry.speakEnd = setTimeout(() => entry.node.classList.remove("is-speaking"), (player.nextTime - now) * 1000);
}

// ------------------------------------------------------------------ Aufnahme

function sendAudio(buffer) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(buffer);
  showLevel(el.level, new Int16Array(buffer));
}

function showLevel(bar, samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / Math.max(1, samples.length)) / 32768;
  const db = 20 * Math.log10(Math.max(rms, 1e-5));
  bar.style.width = `${Math.max(0, Math.min(1, (db + 55) / 50)) * 100}%`;
}

// Öffnet das gewählte Mikrofon und liefert PCM-Blöcke (16 Bit, mono) in der
// gewünschten Abtastrate. Gibt { stop() } zurück oder null, wenn es nicht klappt.
async function openMicrophone({ targetRate, block, onChunk }) {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("Der Browser gibt das Mikrofon nur über https oder localhost frei.");
    return null;
  }
  const deviceId = el.device.value;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        // Die Echounterdrückung hält die Sprachausgabe aus dem Mikrofon heraus.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (error) {
    toast(error.name === "NotAllowedError"
      ? "Der Zugriff auf das Mikrofon wurde nicht erlaubt."
      : error.name === "NotFoundError" || error.name === "OverconstrainedError"
        ? "Das gewählte Mikrofon wurde nicht gefunden."
        : `Mikrofon konnte nicht geöffnet werden: ${error.message}`);
    return null;
  }

  const context = new AudioContext({ latencyHint: "interactive" });
  await context.audioWorklet.addModule("/static/pcm-worklet.js");
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, "pcm-capture", {
    channelCount: 1,
    channelCountMode: "explicit",
    processorOptions: { targetRate, block },
  });
  // Stumm an den Ausgang hängen, damit der Browser den Knoten durchrechnet.
  const mute = context.createGain();
  mute.gain.value = 0;
  node.port.onmessage = (event) => onChunk(event.data);
  source.connect(node).connect(mute).connect(context.destination);
  listDevices(); // Gerätenamen gibt der Browser erst nach der Freigabe heraus

  return {
    stop() {
      node.port.onmessage = null;
      stream.getTracks().forEach((track) => track.stop());
      context.close();
    },
  };
}

function setRunning(kind) {
  const running = Boolean(kind);
  el.toggle.setAttribute("aria-pressed", String(running));
  el.toggleLabel.textContent = running ? "Stopp" : "Start";
  if (!running) el.level.style.width = "0";
  el.vad.dataset.state = running ? "silence" : "off";
  el.vad.textContent = !running ? "Mikrofon aus" : kind === "file" ? "Datei läuft" : "Stille";
  refreshToggle();
}

async function startMic() {
  ensurePlayer();
  const mic = await openMicrophone({ targetRate: SAMPLE_RATE, block: BLOCK, onChunk: sendAudio });
  if (!mic) return;
  capture = mic;
  setRunning("mic");
}

// Liest eine Audiodatei und rechnet sie in PCM (16 Bit, mono) der gewünschten Rate um.
async function decodeFile(file, rate, { padSeconds = 0, maxSeconds = Infinity } = {}) {
  const decoder = new AudioContext();
  const decoded = await decoder.decodeAudioData(await file.arrayBuffer());
  decoder.close();
  const seconds = Math.min(decoded.duration, maxSeconds);
  const offline = new OfflineAudioContext(1, Math.ceil(seconds * rate), rate);
  const input = offline.createBufferSource();
  input.buffer = decoded;
  input.connect(offline.destination);
  input.start();
  const mono = (await offline.startRendering()).getChannelData(0);
  const pcm = new Int16Array(mono.length + Math.round(padSeconds * rate));
  for (let i = 0; i < mono.length; i++) {
    const value = Math.max(-1, Math.min(1, mono[i]));
    pcm[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return pcm;
}

// Spielt eine Audiodatei im Echtzeit-Takt in dieselbe Pipeline wie das Mikrofon.
async function startFile(file) {
  ensurePlayer();
  let pcm;
  try {
    // Eine Sekunde Stille anhängen, damit auch der letzte Satz abgeschlossen wird.
    pcm = await decodeFile(file, SAMPLE_RATE, { padSeconds: 1 });
  } catch (error) {
    toast(`Die Datei konnte nicht gelesen werden: ${error.message}`);
    return;
  }

  let position = 0;
  const startedAt = performance.now();
  const timer = setInterval(() => {
    const due = ((performance.now() - startedAt) / 1000) * SAMPLE_RATE;
    while (position < due && position < pcm.length) {
      sendAudio(pcm.slice(position, position + BLOCK).buffer);
      position += BLOCK;
    }
    if (position >= pcm.length) stopCapture(true);
  }, 30);

  capture = { stop() { clearInterval(timer); } };
  setRunning("file");
}

function stopCapture(tellServer) {
  if (!capture) return;
  capture.stop();
  capture = null;
  if (tellServer) sendJson({ type: "stop" });
  setRunning(null);
}

async function listDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications");
  const fill = (select, kind, storeKey, defaultLabel, fallbackLabel) => {
    const wanted = select.value || store.get(storeKey, "");
    select.replaceChildren(new Option(defaultLabel, ""));
    devices.filter((d) => d.kind === kind).forEach((device, index) => {
      select.add(new Option(device.label || `${fallbackLabel} ${index + 1}`, device.deviceId));
    });
    select.value = [...select.options].some((o) => o.value === wanted) ? wanted : "";
  };
  fill(el.device, "audioinput", "device", "Standardmikrofon", "Mikrofon");
  fill(el.output, "audiooutput", "output", "Standardausgabe", "Ausgabe");

  // VB-Cable: beim ersten Mal automatisch wählen; fehlt es, Einrichtung anbieten.
  const cable = [...el.output.options].find((option) => CABLE.test(option.text));
  if (cable && store.get("output", null) === null) {
    el.output.value = cable.value;
    store.set("output", cable.value);
    toast("VB-Cable gefunden: Die Übersetzung geht jetzt an „CABLE Input“. In Discord oder TeamSpeak „CABLE Output“ als Mikrofon wählen. Mit „Mithören“ hörst du sie zusätzlich selbst.");
  }
  if (!cable && canPickOutput) el.output.add(new Option("Virtuelles Mikrofon einrichten …", SETUP_CABLE));
  outputChanged(false);
  // Ein gemerktes Ausgabegerät taucht erst nach der Mikrofon-Freigabe in der Liste auf.
  if (player.context && canPickOutput && player.context.sinkId !== el.output.value) applyOutput();
}

// ------------------------------------------------------------------- Verlauf

const formatSeconds = (ms) =>
  `${(ms / 1000).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} s`;

function nearBottom() {
  return el.rows.scrollHeight - el.rows.scrollTop - el.rows.clientHeight < 90;
}

function row(id, create = true) {
  const key = `${connection}:${id}`;
  let entry = rows.get(key);
  if (!entry && create) {
    const node = document.createElement("article");
    node.className = "row";
    node.innerHTML = '<p class="src"></p><div><p class="dst"></p><p class="meta"></p></div>';
    entry = { key, node, src: node.querySelector(".src"), dst: node.querySelector(".dst"), meta: node.querySelector(".meta") };
    rows.set(key, entry);
    el.rows.append(node);
    el.empty.hidden = true;
  }
  return entry;
}

function dropRow(id) {
  const entry = row(id, false);
  if (!entry) return;
  entry.node.remove();
  rows.delete(entry.key);
  el.empty.hidden = rows.size > 0;
}

// Zeile unter der Übersetzung: Gesamtverzögerung und die Zeiten der Stufen.
function renderMeta(entry) {
  const total = document.createElement("b");
  const parts = [];
  if (entry.heardMs !== undefined) {
    total.textContent = formatSeconds(entry.heardMs);
    parts.push(" bis zur Sprachausgabe");
  } else {
    total.textContent = formatSeconds(entry.latencyMs);
    parts.push(" bis zur Übersetzung");
  }
  parts.push(`Erkennung ${entry.sttMs} ms`, `Übersetzung ${entry.mtMs} ms`);
  if (entry.ttsMs !== undefined) parts.push(`Sprache ${entry.ttsMs} ms`);
  if (entry.expression !== undefined) parts.push(`Ausdruck ${Math.round(entry.expression * 100)} %`);
  entry.meta.replaceChildren(total, parts.join(" · "));
}

function updateStats() {
  const values = [...delays.values()];
  el.statCount.textContent = String(values.length);
  if (!values.length) {
    el.statLast.textContent = el.statAvg.textContent = "–";
    return;
  }
  el.statLast.textContent = formatSeconds(values.at(-1));
  el.statAvg.textContent = formatSeconds(values.reduce((a, b) => a + b, 0) / values.length);
}

function handle(message) {
  const stick = nearBottom();
  switch (message.type) {
    case "status":
      applyStatus(message.status, message.tts_engine, message.stt_model);
      return;
    case "speech_start":
      row(message.id).node.classList.add("is-live");
      if (capture) { el.vad.dataset.state = "speech"; el.vad.textContent = "Sprache erkannt"; }
      break;
    case "speech_end": {
      const entry = row(message.id);
      entry.node.classList.remove("is-live");
      entry.node.classList.add("is-working");
      if (capture) { el.vad.dataset.state = "silence"; el.vad.textContent = "Stille"; }
      break;
    }
    case "partial": {
      const entry = row(message.id, false);
      if (!entry) return;
      entry.src.textContent = message.text;
      entry.src.classList.add("is-partial");
      break;
    }
    case "transcript": {
      const entry = row(message.id);
      entry.src.textContent = message.text;
      entry.src.classList.remove("is-partial");
      entry.sttMs = message.stt_ms;
      break;
    }
    case "translation_delta":
      row(message.id).dst.textContent += message.delta;
      break;
    case "translation": {
      const entry = row(message.id);
      entry.node.classList.remove("is-working");
      entry.dst.textContent = message.text;
      entry.latencyMs = message.latency_ms;
      entry.mtMs = message.mt_ms;
      delays.set(entry.key, message.latency_ms);
      renderMeta(entry);
      updateStats();
      break;
    }
    case "audio":
      // Der zugehörige Ton kommt als nächste, binäre Nachricht.
      player.header = { ...message, key: `${connection}:${message.id}` };
      return;
    case "audio_end": {
      const entry = row(message.id, false);
      if (!entry) return;
      entry.ttsMs = message.tts_ms;
      entry.expression = message.expression;
      renderMeta(entry);
      return;
    }
    case "discard":
      dropRow(message.id);
      if (capture && el.vad.dataset.state === "speech") { el.vad.dataset.state = "silence"; el.vad.textContent = "Stille"; }
      return;
    case "error": {
      const entry = message.id ? row(message.id, false) : null;
      if (entry) {
        entry.node.classList.remove("is-working");
        entry.meta.textContent = message.message;
        entry.meta.classList.add("is-error");
      } else {
        toast(message.message);
      }
      break;
    }
    default:
      return;
  }
  if (stick) el.rows.scrollTop = el.rows.scrollHeight;
}

// -------------------------------------------------------------------- Stimmen

const VOICE_RATE = 24000;
const VOICE_MIN_SECONDS = 5;
const VOICE_MAX_SECONDS = 25;
const voiceDraft = { mic: null, chunks: [], samples: 0, wav: null, url: "" };

function fillVoices(voices, selected) {
  el.voice.replaceChildren(new Option("Standardstimme", ""));
  for (const voice of voices) el.voice.add(new Option(voice.name, voice.name));
  el.voice.value = voices.some((v) => v.name === selected) ? selected : "";
  voiceChanged(false);
}

function voiceChanged(save = true) {
  el.voiceDelete.disabled = !el.voice.value || !el.speak.checked;
  if (!save) return;
  store.set("voice", el.voice.value);
  sendConfig();
}

function voiceStatus(text, isError = false) {
  el.voiceStatus.textContent = text;
  el.voiceStatus.classList.toggle("is-error", isError);
}

function encodeWav(pcm, rate) {
  const view = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, pcm.length * 2, true);
  new Int16Array(view.buffer, 44).set(pcm);
  return new Blob([view.buffer], { type: "audio/wav" });
}

function showVoiceTime(seconds) {
  el.voiceTime.textContent = `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

// Übernimmt eine fertige Aufnahme: zum Anhören anbieten und Speichern freigeben.
function setVoiceDraft(pcm) {
  const seconds = pcm.length / VOICE_RATE;
  if (voiceDraft.url) URL.revokeObjectURL(voiceDraft.url);
  voiceDraft.wav = encodeWav(pcm, VOICE_RATE);
  voiceDraft.url = URL.createObjectURL(voiceDraft.wav);
  el.voicePreview.src = voiceDraft.url;
  el.voicePreview.hidden = false;
  showVoiceTime(seconds);
  const longEnough = seconds >= VOICE_MIN_SECONDS;
  el.voiceSave.disabled = !longEnough;
  voiceStatus(longEnough
    ? "Hör dir die Aufnahme an. Wenn sie sauber klingt, speichere sie."
    : `Die Aufnahme ist zu kurz, nötig sind mindestens ${VOICE_MIN_SECONDS} Sekunden.`, !longEnough);
}

function resetVoiceDraft() {
  stopVoiceRecording(false);
  if (voiceDraft.url) URL.revokeObjectURL(voiceDraft.url);
  Object.assign(voiceDraft, { chunks: [], samples: 0, wav: null, url: "" });
  el.voicePreview.pause();
  el.voicePreview.removeAttribute("src");
  el.voicePreview.hidden = true;
  el.voiceSave.disabled = true;
  el.voiceLevel.style.width = "0";
  showVoiceTime(0);
  voiceStatus("");
}

async function startVoiceRecording() {
  el.voicePreview.pause();
  Object.assign(voiceDraft, { chunks: [], samples: 0 });
  const mic = await openMicrophone({
    targetRate: VOICE_RATE,
    block: 2400,
    onChunk(buffer) {
      const chunk = new Int16Array(buffer);
      voiceDraft.chunks.push(chunk);
      voiceDraft.samples += chunk.length;
      showLevel(el.voiceLevel, chunk);
      showVoiceTime(voiceDraft.samples / VOICE_RATE);
      if (voiceDraft.samples >= VOICE_MAX_SECONDS * VOICE_RATE) stopVoiceRecording(true);
    },
  });
  if (!mic) {
    voiceStatus("Das Mikrofon konnte nicht geöffnet werden.", true);
    return;
  }
  voiceDraft.mic = mic;
  el.voiceRec.setAttribute("aria-pressed", "true");
  el.voiceRecLabel.textContent = "Fertig";
  el.voiceSave.disabled = true;
  voiceStatus("Aufnahme läuft – lies den Text vor.");
}

function stopVoiceRecording(keep) {
  if (!voiceDraft.mic) return;
  voiceDraft.mic.stop();
  voiceDraft.mic = null;
  el.voiceRec.setAttribute("aria-pressed", "false");
  el.voiceRecLabel.textContent = "Neu aufnehmen";
  el.voiceLevel.style.width = "0";
  if (!keep) return;
  const pcm = new Int16Array(voiceDraft.samples);
  let offset = 0;
  for (const chunk of voiceDraft.chunks) { pcm.set(chunk, offset); offset += chunk.length; }
  setVoiceDraft(pcm);
}

async function saveVoice() {
  const name = el.voiceName.value.trim();
  if (!name) { voiceStatus("Gib der Stimme einen Namen.", true); return; }
  el.voiceSave.disabled = true;
  voiceStatus("Wird gespeichert …");
  try {
    const response = await fetch(`/api/voices/${encodeURIComponent(name)}`, { method: "POST", body: voiceDraft.wav });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Speichern fehlgeschlagen.");
    fillVoices(result.voices, result.name);
    voiceChanged();
    el.dialog.close();
  } catch (error) {
    voiceStatus(error.message, true);
    el.voiceSave.disabled = false;
  }
}

async function deleteVoice() {
  const name = el.voice.value;
  if (!name || !confirm(`Die Stimme „${name}“ löschen?`)) return;
  try {
    const response = await fetch(`/api/voices/${encodeURIComponent(name)}`, { method: "DELETE" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Löschen fehlgeschlagen.");
    fillVoices(result.voices, "");
    voiceChanged();
  } catch (error) {
    toast(error.message);
  }
}

// -------------------------------------------------------- Sprachen und Modelle

function fillLanguages(defaults) {
  el.source.add(new Option("Automatisch erkennen", "auto"));
  for (const language of languages) {
    el.source.add(new Option(language.label, language.code));
    el.target.add(new Option(language.label, language.code));
  }
  const pick = (select, stored, fallback) => {
    select.value = [...select.options].some((o) => o.value === stored) ? stored : fallback;
  };
  pick(el.source, store.get("source", defaults.source), defaults.source);
  pick(el.target, store.get("target", defaults.target), defaults.target);
  languageChanged(false);
}

function languageChanged(save = true) {
  el.swap.disabled = el.source.value === "auto";
  el.headSource.textContent = el.source.selectedOptions[0].textContent;
  el.headTarget.textContent = el.target.selectedOptions[0].textContent;
  if (!save) return;
  store.set("source", el.source.value);
  store.set("target", el.target.value);
  sendConfig();
}

function fillTts(tts) {
  ttsAvailable = tts.enabled;
  for (const engine of tts.engines) {
    const option = new Option(engine.installed ? engine.label : `${engine.label} (nicht eingerichtet)`, engine.id);
    option.disabled = !engine.installed;
    el.engine.add(option);
  }
  el.engine.value = tts.engine;
  fillVoices(tts.voices, store.get("voice", tts.voice));
  el.speak.checked = ttsAvailable && store.get("speak", "true") === "true";
  el.volume.value = store.get("volume", "100");
  el.expression.value = store.get("expression", String(Math.round(tts.expression * 100)));
  el.expressionOut.textContent = `${el.expression.value} %`;
  el.expressionAuto.checked = store.get("expressionAuto", "true") === "true";
  applyVolume();
  speakChanged(false);
  if (!ttsAvailable) {
    el.speak.disabled = true;
    el.speak.closest(".switch").title = "Die Sprachausgabe ist in config.yaml ausgeschaltet.";
  }
  if (!canPickOutput) {
    el.output.title = "Dieser Browser kann das Ausgabegerät nicht umstellen.";
  }
}

function outputChanged(save = true) {
  el.monitor.disabled = !el.output.value || !el.speak.checked;
  ensureMonitor();
  if (save) {
    store.set("output", el.output.value);
    applyOutput();
  }
}

function speakChanged(save = true) {
  const on = el.speak.checked;
  el.output.disabled = !on || !canPickOutput;
  el.monitor.disabled = !on || !el.output.value;
  ensureMonitor();
  el.engine.disabled = !on;
  el.volume.disabled = !on;
  el.expression.disabled = !on;
  el.expressionAuto.disabled = !on;
  el.voice.disabled = !on;
  el.voiceNew.disabled = !on;
  el.voiceDelete.disabled = !on || !el.voice.value;
  if (!on) stopPlayback();
  if (!save) return;
  store.set("speak", String(on));
  if (on) ensurePlayer();
  sendConfig();
}

// ----------------------------------------------------------------------- Start

function bind() {
  el.toggle.addEventListener("click", () => (capture ? stopCapture(true) : startMic()));

  el.file.addEventListener("change", () => {
    const file = el.file.files[0];
    el.file.value = "";
    if (!file) return;
    stopCapture(true);
    startFile(file);
  });

  el.device.addEventListener("change", () => {
    store.set("device", el.device.value);
    if (capture) { stopCapture(true); startMic(); }
  });

  el.source.addEventListener("change", () => languageChanged());
  el.target.addEventListener("change", () => languageChanged());
  el.swap.addEventListener("click", () => {
    [el.source.value, el.target.value] = [el.target.value, el.source.value];
    languageChanged();
  });

  el.pause.addEventListener("input", () => { el.pauseOut.textContent = `${el.pause.value} ms`; });
  el.pause.addEventListener("change", () => { store.set("pause", el.pause.value); sendConfig(); });
  el.style.addEventListener("change", () => { store.set("style", el.style.value); sendConfig(); });
  el.sttModel.addEventListener("change", () => sendJson({ type: "stt_model", model: el.sttModel.value }));

  el.speak.addEventListener("change", () => speakChanged());
  el.output.addEventListener("change", () => {
    if (el.output.value === SETUP_CABLE) {
      // VB-Cable ist ein Treiber und muss einmal von Hand installiert werden.
      window.open(CABLE_PAGE, "_blank", "noopener");
      el.output.value = store.get("output", "");
      toast("VB-Cable installieren, den Rechner neu starten und RTlation neu laden. Danach erscheint „CABLE Input“ in dieser Liste.");
      return;
    }
    outputChanged();
  });
  el.monitor.checked = store.get("monitor", "true") === "true";
  el.monitor.addEventListener("change", () => { store.set("monitor", String(el.monitor.checked)); ensureMonitor(); });
  el.engine.addEventListener("change", () => sendJson({ type: "tts_engine", engine: el.engine.value }));
  el.volume.addEventListener("input", applyVolume);
  el.volume.addEventListener("change", () => store.set("volume", el.volume.value));

  el.expression.addEventListener("input", () => { el.expressionOut.textContent = `${el.expression.value} %`; });
  el.expression.addEventListener("change", () => { store.set("expression", el.expression.value); sendConfig(); });
  el.expressionAuto.addEventListener("change", () => { store.set("expressionAuto", String(el.expressionAuto.checked)); sendConfig(); });
  el.voice.addEventListener("change", () => voiceChanged());
  el.voiceDelete.addEventListener("click", deleteVoice);
  el.voiceNew.addEventListener("click", () => {
    stopCapture(true);
    resetVoiceDraft();
    el.voiceRecLabel.textContent = "Aufnehmen";
    el.dialog.showModal();
  });
  el.voiceRec.addEventListener("click", () => (voiceDraft.mic ? stopVoiceRecording(true) : startVoiceRecording()));
  el.voiceFile.addEventListener("change", async () => {
    const file = el.voiceFile.files[0];
    el.voiceFile.value = "";
    if (!file) return;
    stopVoiceRecording(false);
    try {
      setVoiceDraft(await decodeFile(file, VOICE_RATE, { maxSeconds: 30 }));
    } catch (error) {
      voiceStatus(`Die Datei konnte nicht gelesen werden: ${error.message}`, true);
    }
  });
  el.voiceSave.addEventListener("click", saveVoice);
  el.voiceCancel.addEventListener("click", () => el.dialog.close());
  el.dialog.addEventListener("close", resetVoiceDraft);

  el.clear.addEventListener("click", () => {
    rows.clear();
    delays.clear();
    el.rows.replaceChildren();
    updateStats();
    el.empty.hidden = false;
  });

  navigator.mediaDevices?.addEventListener?.("devicechange", listDevices);
}

async function init() {
  buildThemes();
  bind();
  let info;
  try {
    info = await (await fetch("/api/info")).json();
  } catch {
    setPill(el.stConn, "error", "Server nicht erreichbar");
    setTimeout(() => location.reload(), 3000);
    return;
  }
  languages = info.languages;
  fillLanguages(info.defaults);
  for (const style of info.styles) el.style.add(new Option(style.label, style.id));
  const wantedStyle = store.get("style", info.defaults.style);
  el.style.value = info.styles.some((s) => s.id === wantedStyle) ? wantedStyle : info.defaults.style;
  fillTts(info.tts);
  el.pause.value = store.get("pause", info.defaults.min_silence_ms);
  el.pauseOut.textContent = `${el.pause.value} ms`;
  for (const model of info.stt.models) el.sttModel.add(new Option(model.label, model.id));
  applyStatus(info.status, info.tts.engine, info.stt.model);
  connect();
  listDevices();
}

init();
