"""Quick mic test — run this locally on the Mac Mini to check audio levels."""

import sounddevice as sd
import numpy as np

print("Recording 5 seconds from USB mic... speak now!")
audio = sd.rec(int(5 * 48000), samplerate=48000, channels=1, dtype="float32", device=0)
sd.wait()

peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))

print(f"Peak: {peak:.4f}")
print(f"RMS:  {rms:.6f}")

if peak < 0.001:
    print("\nNO AUDIO — mic is not capturing sound.")
    print("Check: System Settings > Privacy & Security > Microphone")
elif peak < 0.05:
    print("\nLow audio — move closer to the mic or speak louder.")
else:
    print("\nMic is working!")
