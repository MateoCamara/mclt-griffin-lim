"""Example: reconstruct audio from magnitude with Griffin-Lim — STFT vs MCLT.

This script loads an audio file, computes both its STFT and its MCLT, discards the
phase (keeping only the magnitude), and then reconstructs a time-domain signal from
each magnitude spectrogram using Griffin-Lim:

* STFT magnitude -> ``librosa.griffinlim``               -> ``test_audio_stft.wav``
* MCLT magnitude -> ``mclt_griffinlim.mclt_griffinlim``  -> ``test_audio_mclt.wav``

Listening to the two outputs lets you compare phase reconstruction on the ordinary
STFT against the Modulated Complex Lapped Transform.

Run it with:  python test.py
"""

import os

import librosa
import mdct
import soundfile as sf

import mclt_griffinlim

SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 512

# Resolve paths relative to this file so the script works from any directory.
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # Load the example audio as a mono signal.
    signal = librosa.load(os.path.join(HERE, "test_audio.wav"),
                          sr=SAMPLE_RATE,
                          mono=True)[0]

    # Complex STFT, dropping the Nyquist bin.
    stft_complex = librosa.stft(signal,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH)[:-1]

    # Complex MCLT, dropping the trailing edge frame to match the magnitude layout.
    mclt_complex = mdct.fast.mclt(signal,
                                  framelength=N_FFT,
                                  hopsize=HOP_LENGTH)[:, :-1]

    # Keep only the magnitudes (discard the phase) for each transform.
    spectrogram_stft, _ = librosa.magphase(stft_complex)
    spectrogram_mclt, _ = librosa.magphase(mclt_complex)

    # Reconstruct a time-domain signal from each magnitude via Griffin-Lim.
    signal_stft = librosa.griffinlim(spectrogram_stft,
                                     hop_length=HOP_LENGTH)
    signal_mclt = mclt_griffinlim.mclt_griffinlim(spectrogram_mclt,
                                                  frame_length=N_FFT)

    # Write the two reconstructions next to this script for comparison.
    sf.write(os.path.join(HERE, "test_audio_stft.wav"), signal_stft, SAMPLE_RATE)
    sf.write(os.path.join(HERE, "test_audio_mclt.wav"), signal_mclt, SAMPLE_RATE)

    print("Wrote test_audio_stft.wav and test_audio_mclt.wav")


if __name__ == "__main__":
    main()
