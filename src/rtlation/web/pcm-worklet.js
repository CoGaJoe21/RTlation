// Wandelt das Mikrofonsignal in PCM (16 Bit, mono) um und reicht es in
// Blöcken an die Seite weiter. Läuft im Audio-Thread des Browsers.
//
// processorOptions: { targetRate, block }
//   16 kHz für die Spracherkennung (Standard), 24 kHz für Stimmaufnahmen.

class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { targetRate = 16000, block = 1024 } = options.processorOptions || {};
    this.ratio = sampleRate / targetRate;
    this.block = new Int16Array(block);
    this.filled = 0;
    this.sum = 0;
    this.count = 0;
    this.position = 0;
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;

    // Herunterrechnen durch Mitteln: Alle Eingangswerte, die auf einen
    // Ausgangswert fallen, werden gemittelt.
    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count++;
      this.position++;
      if (this.position < this.ratio) continue;

      this.position -= this.ratio;
      const value = Math.max(-1, Math.min(1, this.sum / this.count));
      this.block[this.filled++] = value < 0 ? value * 0x8000 : value * 0x7fff;
      this.sum = 0;
      this.count = 0;

      if (this.filled === this.block.length) {
        const out = this.block.buffer.slice(0);
        this.port.postMessage(out, [out]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
