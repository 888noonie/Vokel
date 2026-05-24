/** Short UI earcons for external tool activity (Web Audio, no samples required). */

const CLICK_ATTACK_S = 0.0015;
const CLICK_DECAY_S = 0.048;
const CLICK_GAP_S = 0.072;

/** Ascending major third — reads as “working”. */
const START_TONES_HZ = [880, 1108.73] as const;

/** Soft descending pair — reads as “done”. */
const END_TONES_HZ = [988, 783.99] as const;

function scheduleClick(
  audioContext: AudioContext,
  frequencyHz: number,
  startTime: number,
  peakGain: number,
): void {
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();

  oscillator.type = "triangle";
  oscillator.frequency.setValueAtTime(frequencyHz, startTime);
  oscillator.frequency.exponentialRampToValueAtTime(
    frequencyHz * 0.72,
    startTime + CLICK_DECAY_S,
  );

  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(peakGain, startTime + CLICK_ATTACK_S);
  gain.gain.exponentialRampToValueAtTime(0.0001, startTime + CLICK_DECAY_S);

  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start(startTime);
  oscillator.stop(startTime + CLICK_DECAY_S + 0.01);
}

function playTonePair(
  audioContext: AudioContext,
  tonesHz: readonly number[],
  peakGain: number,
): void {
  const baseTime = audioContext.currentTime + 0.005;
  tonesHz.forEach((frequencyHz, index) => {
    scheduleClick(
      audioContext,
      frequencyHz,
      baseTime + index * CLICK_GAP_S,
      peakGain * (index === 0 ? 1 : 0.88),
    );
  });
}

export async function playToolStartClick(audioContext: AudioContext): Promise<void> {
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  playTonePair(audioContext, START_TONES_HZ, 0.055);
}

export async function playToolEndClick(audioContext: AudioContext): Promise<void> {
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  playTonePair(audioContext, END_TONES_HZ, 0.038);
}
