import os, wave, csv, re
import numpy as np
from deepspeech import Model

BASE = "/work"

MODELS = {
    "EN": {
        "model": os.path.join(BASE, "EN", "deepspeech-0.9.3-models.pbmm"),
        "scorer": os.path.join(BASE, "EN", "deepspeech-0.9.3-models.scorer"),
    },
    "ES": {
        "model": os.path.join(BASE, "ES", "model.pbmm"),
        "scorer": os.path.join(BASE, "ES", "kenlm.scorer"),
    },
    "IT": {
        "model": os.path.join(BASE, "IT", "model.pbmm"),
        "scorer": os.path.join(BASE, "IT", "kenlm.scorer"),
    },
}

# ---------------------------
# Reference transcriptions
# (from Ex4_audio_files_transcriptions.pdf)
# ---------------------------
REF = {
    # English
    "checkin.wav": "Where is the check-in desk?",
    "parents.wav": "I have lost my parents.",
    "suitcase.wav": "Please, I have lost my suitcase.",
    "what_time.wav": "What time is my plane?",
    "where.wav": "Where are the restaurants and shops?",
    "your_sentence1.wav": "I don't want anything to drink i'm rather hungry.",
    "your_sentence2.wav": "I excercise for an hour every morning.",

    # Spanish
    "checkin_es.wav": "¿Dónde están los mostradores?",
    "parents_es.wav": "He perdido a mis padres.",
    "suitcase_es.wav": "Por favor, he perdido mi maleta.",
    "what_time_es.wav": "¿A qué hora es mi avión?",
    "where_es.wav": "¿Dónde están los restaurantes y las tiendas?",

    # Italian
    "checkin_it.wav": "Dove è il bancone?",
    "parents_it.wav": "Ho perso i miei genitori.",
    "suitcase_it.wav": "Per favore, ho perso la mia valigia.",
    "what_time_it.wav": "A che ora è il mio aereo?",
    "where_it.wav": "Dove sono i ristoranti e i negozi?",
}

# ---------------------------
# Audio loading
# ---------------------------
def read_wav_16k_mono(path):
    # Reads a WAV file, ensures mono 16-bit PCM, and resamples to 16kHz if needed.
    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        audio_bytes = wf.readframes(n)

    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={sw}")

    x = np.frombuffer(audio_bytes, dtype=np.int16)

    # If stereo/multichannel → convert to mono
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)

    # If not 16kHz → resample to 16kHz (so all WAVs work, incl. 48kHz recordings)
    if sr != 16000:
        from scipy.signal import resample_poly
        xf = x.astype(np.float32)
        y = resample_poly(xf, 16000, sr)
        x = np.clip(y, -32768, 32767).astype(np.int16)

    return x
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\wàèéìòóùüñç\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def wer(ref, hyp):
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()

    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m] / max(1, n)

# ---------------------------
# Main
# ---------------------------
def main():
    ds = {}
    for lang, mp in MODELS.items():
        print(f"Loading {lang} model...")
        ds[lang] = Model(mp["model"])
        ds[lang].enableExternalScorer(mp["scorer"])
        print("OK")

    rows = []

    for lang in ["EN", "ES", "IT"]:
        folder = os.path.join(BASE, lang)
        wavs = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))

        for w in wavs:
            audio = read_wav_16k_mono(os.path.join(folder, w))
            hyp = ds[lang].stt(audio)
            ref = REF.get(w, "")
            file_wer = wer(ref, hyp) if ref else None

            print(f"{lang}/{w}")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")
            print(f"  WER: {file_wer}\n")

            rows.append([
                lang,
                w,
                ref,
                hyp,
                file_wer
            ])

    out_csv = os.path.join(BASE, "ex4_predictions_with_wer.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "file", "reference", "prediction", "wer"])
        writer.writerows(rows)

    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    main()
