# mclt-griffin-lim

**Fast Griffin-Lim phase reconstruction for the Modulated Complex Lapped Transform (MCLT).**

A version of the Griffin-Lim algorithm prepared to operate on magnitude spectrograms
of the **Modulated Complex Lapped Transform (MCLT)** instead of the usual STFT. Given
only the MCLT *magnitude*, it recovers a compatible phase and reconstructs a
real-valued time-domain signal.

## What & why

The **MCLT** (Malvar, 1999) is a complex extension of the MDCT: a critically sampled,
perfectly reconstructing *lapped* transform whose complex coefficients carry both a
magnitude and a meaningful phase, while its overlapping windows reduce the block
artifacts of a plain DFT frame. It is an attractive front-end for audio because of its
good time/frequency localization and perfect reconstruction.

**Griffin-Lim** recovers a signal from a magnitude spectrogram when the phase is
unknown (or has been modified). This repository ports the **"fast" Griffin-Lim**
algorithm (Perraudin et al., 2013) — as found in `librosa.griffinlim` for the STFT — so
that it iterates inverse/forward **MCLT** operations instead of inverse/forward STFT.

## Algorithm

On each iteration the algorithm:

1. takes the inverse MCLT of `magnitude × current_phase` to get a time-domain signal,
2. takes the forward MCLT of that signal to obtain a new spectral estimate,
3. applies the fast-GL momentum extrapolation, and
4. discards the new magnitude (renormalising to unit magnitude), keeping only the
   updated phase.

After `n_iter` iterations, a final inverse MCLT returns the reconstructed signal.

## Installation

> ⚠️ **Requires Python 3.8–3.10.** The MCLT/STFT backends are abandonware and only run
> on a pre-2023 scientific-Python stack.

```bash
# from a fresh Python 3.10 (or 3.8/3.9) virtual environment
pip install -r requirements.txt
```

The MCLT/STFT backends are the (now unmaintained) packages by **Nils Werner** —
[`mdct`](https://github.com/nils-werner/mdct) and
[`stft`](https://github.com/nils-werner/stft). The `requirements.txt` pins an exact,
end-to-end-verified stack because of three hard constraints they impose:

- **`numpy < 1.23`** — `stft` indexes arrays with a *list* of slices, which NumPy turned
  into an error in 1.23. (numpy 1.22 also has no wheels for Python 3.11+, hence the
  Python ≤ 3.10 requirement.)
- **`scipy < 1.9`** — `stft` uses `scipy.real`, removed from the top-level `scipy`
  namespace in SciPy 1.9.
- **`scipy < 1.13`** — `mdct` imports `scipy.signal.kaiser`, removed in SciPy 1.13
  (already covered by `scipy < 1.9`).

> Tip: always install into a fresh virtual environment to keep this legacy stack
> isolated from the rest of your setup.

## Usage

```python
import librosa, mdct
import mclt_griffinlim

# Load audio and compute its MCLT (frequency × frames).
signal = librosa.load("test_audio.wav", sr=44100, mono=True)[0]
mclt = mdct.fast.mclt(signal, framelength=1024, hopsize=512)[:, :-1]

# Keep only the magnitude (this is what we will invert).
magnitude, _ = librosa.magphase(mclt)

# Reconstruct a time-domain signal from the magnitude alone.
reconstruction = mclt_griffinlim.mclt_griffinlim(
    magnitude, frame_length=1024, n_iter=32, random_state=0
)
```

### Run the example

`test.py` reconstructs the bundled `test_audio.wav` from magnitude with **both** STFT
and MCLT Griffin-Lim, so you can compare them:

```bash
python test.py
# writes test_audio_stft.wav and test_audio_mclt.wav
```

## Parameters

| Parameter      | Default    | Description |
|----------------|------------|-------------|
| `S`            | —          | MCLT **magnitude**, shape `(frequency_subbands, time_frames)`. |
| `n_iter`       | `32`       | Number of Griffin-Lim iterations. |
| `frame_length` | `None`     | MCLT frame length; **must be a power of 2**. Pass it explicitly (inference may not be a power of 2). |
| `sample_rate`  | `44100`    | Currently unused; kept for backward compatibility. |
| `center`       | `True`     | Centered vs. left-aligned frames. |
| `momentum`     | `0.99`     | Fast-GL momentum; `0` recovers classic Griffin-Lim, `>1` may diverge. |
| `init`         | `'random'` | `'random'` phase init, or `None` for zero (deterministic) phase. |
| `random_state` | `None`     | Seed (int) or `RandomState` for reproducible output. |

**Input layout:** frequency along the first axis, time along the second — the
orientation produced by `mdct.fast.mclt`.

## STFT vs MCLT

`librosa` already ships STFT Griffin-Lim; this repo adds the MCLT variant. The example
writes one reconstruction from each so you can listen to the difference in how phase is
recovered on a lapped complex transform versus the STFT.

## References

- H. S. Malvar, *"A modulated complex lapped transform and its applications to audio
  processing,"* ICASSP, 1999.
- D. W. Griffin and J. S. Lim, *"Signal estimation from modified short-time Fourier
  transform,"* IEEE Trans. ASSP, vol. 32, no. 2, pp. 236–243, 1984.
- N. Perraudin, P. Balazs, and P. L. Søndergaard, *"A fast Griffin-Lim algorithm,"*
  IEEE WASPAA, 2013.

The iteration is adapted from [`librosa.griffinlim`](https://github.com/librosa/librosa).

## License

Released under the [MIT License](LICENSE).

---

If you find this useful, please give it a ⭐. Comments and ideas are welcome — contact
details at <https://www.mateocamara.com>.
