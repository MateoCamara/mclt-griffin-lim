import soundfile as sf
import mdct
import librosa

import mclt_griffinlim

signal = librosa.load("test_audio.wav",
                      sr=44100,
                      mono=True)[0]

stft = librosa.stft(signal,
                    n_fft=1024,
                    hop_length=512)[:-1]

mclt = mdct.fast.mclt(signal,
                      framelength=1024,
                      hopsize=512)[:, :-1]

spectrogram_stft, phase_stft = librosa.magphase(stft)
spectrogram_mclt, phase_mclt = librosa.magphase(mclt)

signal_stft = librosa.griffinlim(spectrogram_stft,
                                 hop_length=512)
signal_mclt = mclt_griffinlim.mclt_griffinlim(spectrogram_mclt,
                                   frame_length=1024)

sf.write("test_audio_stft.wav", signal_stft, 44100)
sf.write("test_audio_mclt.wav", signal_mclt, 44100)

print('end')

