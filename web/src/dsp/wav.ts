/**
 * Encode stereo audio as a 32-bit float WAV and return a blob URL.
 *
 * Format: RIFF/WAVE, format tag 3 (IEEE float), 44-byte header.
 * Caller is responsible for revoking the URL via URL.revokeObjectURL().
 */
export function encodeWavUrl(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
): string {
  const numSamples = left.length;
  const numChannels = 2;
  const bytesPerSample = 4; // 32-bit float
  const dataSize = numSamples * numChannels * bytesPerSample;
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, headerSize - 8 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 3, true); // format tag: IEEE float
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
  view.setUint16(32, numChannels * bytesPerSample, true); // block align
  view.setUint16(34, bytesPerSample * 8, true); // bits per sample

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Interleaved samples: [L0, R0, L1, R1, ...]
  // Float32Array is ~5-10x faster than DataView.setFloat32 for large buffers.
  // Note: Float32Array uses platform endianness (little-endian on all modern JS runtimes),
  // which matches WAV format.
  const data = new Float32Array(buffer, headerSize, numSamples * numChannels);
  for (let i = 0; i < numSamples; i++) {
    data[i * 2] = left[i];
    data[i * 2 + 1] = right[i];
  }

  const blob = new Blob([buffer], { type: "audio/wav" });
  return URL.createObjectURL(blob);
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
