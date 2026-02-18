---
metadata:
  name: "librispeech-asr-test-clips"
  version: "1.0.0"
  description: "Download LibriSpeech test-clean data and produce named audio + transcript pairs of configurable lengths for ASR quality and WER testing"
  category: "asr-evaluation"
  tags: ["asr", "wer", "librispeech", "audio", "speech", "benchmark", "nlp"]
  author: "your-github-username"
---

# Create LibriSpeech ASR Test Audio Clips

## Overview

This skill produces ready-to-use audio + transcript pairs from the
[LibriSpeech test-clean](https://www.openslr.org/12) corpus — the standard benchmark for
English ASR evaluation. A single Parquet file (~350 MB) is downloaded from HuggingFace; a
script then decodes the embedded FLAC bytes and concatenates utterances into clips of any
desired length. The output is plain WAV + TXT pairs that can be fed directly into any ASR
pipeline for WER measurement.

Default output clips: **5 min**, **15 min**, and **30 min** — all 16 kHz mono WAV with one
reference sentence per line in the matching transcript file.

---

## Steps

### 1. Create a Virtual Environment and Install Dependencies


python3 -m venv .venv
source .venv/bin/activate
pip install pyarrow requests librosa soundfile numpy


Creates an isolated Python environment and installs the required packages:

| Package | Purpose |
|---------|---------|
| `pyarrow` | Read the `.parquet` dataset file |
| `librosa` | Decode FLAC audio bytes into numpy arrays |
| `soundfile` | Write the final `.wav` files |
| `numpy` | Concatenate audio arrays |

### 2. Create the Output Directory


mkdir -p asr_test


All downloaded and generated files will live here.

### 3. Download the LibriSpeech test-clean Parquet File


curl -L \
  "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet" \
  -o asr_test/test_clean.parquet


Downloads approximately 350 MB. This single file contains all 2,620 test-clean utterances
(~5.4 hours of audio total) with audio bytes and reference transcripts bundled together —
no extraction step needed.

> **If `curl` is unavailable**, use `wget` instead:
> 
> wget -O asr_test/test_clean.parquet \
>   "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet"
> 

### 4. Run the Extraction Script

Save the following as `asr_test/extract_clips.py`, then run it:


python3 asr_test/extract_clips.py



"""
extract_clips.py — build named audio + transcript clips from LibriSpeech parquet.
Edit TARGETS to change clip lengths. Add more entries for longer clips.
"""
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import io, os, librosa

# ── Configuration ─────────────────────────────────────────────────────────────
PARQUET_PATH = "asr_test/test_clean.parquet"
OUTPUT_DIR   = "asr_test"

TARGETS = {
    "5min":  5  * 60,   # seconds
    "15min": 15 * 60,
    "30min": 30 * 60,
}
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading parquet …")
table = pq.read_table(PARQUET_PATH)
data  = table.to_pydict()
print(f"  {len(data['audio'])} utterances available")

clips = {k: {"samples": [], "transcripts": [], "dur": 0.0} for k in TARGETS}
done  = set()

print("Decoding audio and building clips …")
for audio_dict, text in zip(data["audio"], data["text"]):
    if len(done) == len(TARGETS):
        break
    raw = audio_dict["bytes"]           # raw FLAC bytes stored in the parquet
    arr, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    dur = len(arr) / sr

    for name, target in TARGETS.items():
        if name in done:
            continue
        c = clips[name]
        c["samples"].append(arr.copy())
        c["transcripts"].append(text)
        c["dur"] += dur
        if c["dur"] >= target:
            done.add(name)
            print(f"  {name} complete — {c['dur']:.1f}s, {len(c['samples'])} utterances")

print("\nWriting output files …")
for name, c in clips.items():
    combined = np.concatenate(c["samples"])
    wav_path = os.path.join(OUTPUT_DIR, f"{name}_audio.wav")
    txt_path = os.path.join(OUTPUT_DIR, f"{name}_transcript.txt")
    sf.write(wav_path, combined, 16000)
    with open(txt_path, "w") as f:
        for t in c["transcripts"]:
            f.write(t + "\n")
    print(f"  {name}: {wav_path}  ({os.path.getsize(wav_path)/1e6:.1f} MB)")

print("\nDone.")


The script iterates over utterances in one pass. For each utterance it appends audio and text
to every clip that has not yet hit its target duration, then stops as soon as all clips are
complete. This avoids loading the entire dataset into memory at once.

> **Why Parquet instead of the full `test-clean.tar.gz`?**  
> The tar.gz is 337 MB and requires extracting thousands of individual FLAC files.
> The HuggingFace Parquet version bundles everything into one file readable row-by-row
> with no extraction step.

---

## Expected Output


Loading parquet …
  2620 utterances available
Decoding audio and building clips …
  5min complete  — 302.9s,  46 utterances
  15min complete — 903.1s,  129 utterances
  30min complete — 1803.2s, 242 utterances

Writing output files …
  5min:  asr_test/5min_audio.wav  (9.7 MB)
  15min: asr_test/15min_audio.wav (28.9 MB)
  30min: asr_test/30min_audio.wav (57.7 MB)

Done.


- `asr_test/5min_audio.wav` — 5 min, 16 kHz mono WAV, 9.7 MB
- `asr_test/5min_transcript.txt` — 46 reference sentences (one per line, UPPERCASE)
- `asr_test/15min_audio.wav` — 15 min, 16 kHz mono WAV, 28.9 MB
- `asr_test/15min_transcript.txt` — 129 reference sentences
- `asr_test/30min_audio.wav` — 30 min, 16 kHz mono WAV, 57.7 MB
- `asr_test/30min_transcript.txt` — 242 reference sentences

---

## Troubleshooting

### Missing packages


pip install pyarrow librosa soundfile numpy


### Download fails with HTTP 401


pip install huggingface_hub
huggingface-cli login


Then re-run the `curl` command in Step 3.

### Want different clip lengths

Edit `TARGETS` in `extract_clips.py`:


TARGETS = {
    "10min": 10 * 60,
    "60min": 60 * 60,
}


### Want noisier / accented speech

Replace `test.clean` with `test.other` in the download URL:


curl -L \
  "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.other/0000.parquet" \
  -o asr_test/test_other.parquet


Then update `PARQUET_PATH` in the script accordingly.

---

## Related Skills

- `asr-wer-evaluation`
- `whisper-inference`
- `openvino-asr-optimization`

---

## References

- [LibriSpeech ASR Corpus — OpenSLR](https://www.openslr.org/12)
- [LibriSpeech on HuggingFace](https://huggingface.co/datasets/openslr/librispeech_asr)
- [LibriSpeech paper (Panayotov et al., 2015)](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
- [soundfile documentation](https://python-soundfile.readthedocs.io)
- [librosa documentation](https://librosa.org/doc/latest/index.html)