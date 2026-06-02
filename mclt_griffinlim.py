"""Fast Griffin-Lim phase reconstruction for the Modulated Complex Lapped Transform (MCLT).

This module adapts the "fast" Griffin-Lim algorithm (Perraudin et al., 2013) — as
implemented in ``librosa.griffinlim`` for the STFT — so that it operates on the
**Modulated Complex Lapped Transform** (MCLT, Malvar 1999) instead of the STFT.

Why the MCLT? The MCLT is a complex extension of the MDCT: a critically sampled,
perfectly reconstructing *lapped* transform whose complex coefficients carry both a
magnitude and a meaningful phase, while its overlapping windows reduce the block
artifacts of a plain DFT frame. Given only the **magnitude** of an MCLT spectrogram,
this function recovers a compatible phase (and therefore a real time-domain signal)
by iterating inverse- and forward-MCLT operations and, on each step, keeping the
estimated phase but forcing the magnitude back to the target ``S``.

The forward/inverse MCLT are provided by the ``mdct`` package and the analysis
window by the ``stft`` package (both by Nils Werner). These are unmaintained and
require ``scipy < 1.13`` (``scipy.signal.kaiser`` was removed in 1.13) and
``numpy < 2``; see ``requirements.txt``.

References
----------
* H. S. Malvar, "A modulated complex lapped transform and its applications to audio
  processing," ICASSP, 1999.
* D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier
  transform," IEEE Trans. ASSP, vol. 32, no. 2, pp. 236-243, 1984.
* N. Perraudin, P. Balazs, and P. L. Søndergaard, "A fast Griffin-Lim algorithm,"
  IEEE WASPAA, 2013.

Adapted from ``librosa.griffinlim`` (ISC License).
"""

import warnings

import librosa
import mdct
import numpy as np
import stft
from librosa import ParameterError


def mclt_griffinlim(
    S,
    *,
    n_iter=32,
    frame_length=None,
    sample_rate=44100,
    center=True,
    momentum=0.99,
    init="random",
    random_state=None,
):

    """Approximate MCLT magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.

    Given a Modulated Complex Lapped Transform magnitude matrix (``S``), the algorithm
    initializes the phase estimates (randomly by default) and then alternates inverse-
    and forward-MCLT operations, keeping the running phase estimate while restoring the
    target magnitude on every iteration. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and that
    ``S`` is laid out as ``(frequency_subbands, time_frames)`` (the orientation produced
    by ``mdct.fast.mclt`` / the ``stft`` package).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    .. [#] H. Malvar,
       "A modulated complex lapped transform and its applications to audio processing,"
        presented at the 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Proceedings. ICASSP99, 1999.

    Parameters
    ----------
    S : np.ndarray [shape=(frequency_subbands, time_frames), non-negative]
        An array of MCLT magnitudes as produced by ``mdct.fast.mclt`` followed by
        ``librosa.magphase`` (i.e. the magnitude part). The expected layout is
        frequency along the first axis and time along the second.

    n_iter : int > 0
        The number of iterations to run.

    frame_length : None or int > 0
        The frame length of the MCLT. It must be a power of 2. If not provided, it
        defaults to ``2 * (S.shape[-2] - 1)`` and a warning is raised, because the
        inferred value may not be a power of 2 — passing it explicitly is recommended.

    sample_rate : int
        Sampling rate of the target signal. Currently unused; retained for backward
        compatibility and possible future use. (It previously fed an incorrect output
        length; see the note in the source.)

    center : boolean
        If ``True``, the MCLT is assumed to use centered frames.
        If ``False``, the MCLT is assumed to use left-aligned frames.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``. This is recommended when the input ``S`` is a
        magnitude spectrogram with no initial phase estimates.

        If ``None``, then the phase is initialized to zero (all-ones complex). This is
        useful when you want a deterministic start or to resume Griffin-Lim from a
        previous output.

    random_state : None, int, or np.random.RandomState
        If int, ``random_state`` is the seed used by the random number generator for
        phase initialization (use this for reproducible output).

        If ``np.random.RandomState`` instance, the random number generator itself.

        If ``None``, defaults to the current ``np.random`` object (non-reproducible).

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        Time-domain signal reconstructed from ``S``.

    See Also
    --------
    librosa
    mdct
    stft
    magphase

    """

    # Resolve the random number generator used for phase initialization.
    rng = np.random
    if isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    # Validate the momentum: > 1 is unstable, < 0 is invalid.
    if momentum > 1:
        warnings.warn(
            "Griffin-Lim with momentum={} > 1 can be unstable. "
            "Proceed with caution!".format(momentum),
            stacklevel=2,
        )
    elif momentum < 0:
        raise ParameterError(
            "griffinlim() called with momentum={} < 0".format(momentum)
        )

    # Infer the frame length from the spectrogram shape if it was not provided.
    # The inferred value is not guaranteed to be a power of 2, hence the warning.
    if frame_length is None:
        frame_length = 2 * (S.shape[-2] - 1)
        warnings.warn(
            "Mclt requires frame_length to be a power of 2, it's better to introduce frame_length manually"
        )

    # The MCLT requires a power-of-two frame length.
    if not ((frame_length & (frame_length-1) == 0) and frame_length != 0):
        raise ParameterError(
            "frame_length is not a power of 2"
        )

    # 50% overlap between consecutive lapped frames.
    hop_length = int(frame_length / 2)

    # Complex phase buffer. complex64 keeps the result to the minimal necessary precision.
    angles = np.empty(S.shape, dtype=np.complex64)
    eps = librosa.util.tiny(angles)

    if init == "random":
        # Random initial phase, uniform on the unit circle.
        angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
    elif init is None:
        # Zero initial phase (all-ones complex matrix).
        angles[:] = 1.0
    else:
        raise ParameterError("init={} must either None or 'random'".format(init))

    # Previous iterate, used by the fast-GL momentum term (starts at 0).
    rebuilt = 0.0

    for _ in range(n_iter):
        # Store the previous reconstruction for the momentum extrapolation.
        tprev = rebuilt

        # Inverse MCLT of (target magnitude * current phase) -> time-domain signal.
        # outlength is left at its default so the ``stft`` backend reconstructs the
        # natural signal length. (The previous code passed
        # ``outlength=int(sample_rate * frame_length / 2)``, which is dimensionally a
        # sample_rate x samples product; it was a no-op for normal audio because the
        # backend only truncates, but it would silently truncate very long inputs.)
        inverse = mdct.fast.imclt(S * angles,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,  # explicit; this is also the stft default
                        padding=0)

        # Forward MCLT back to the spectral domain. The trailing column is dropped so
        # the round-trip shape matches the target ``S`` (the edge frame the lapped
        # transform adds on reconstruction).
        rebuilt = mdct.fast.mclt(inverse,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,
                        padding=0)[:, :-1]

        # Fast Griffin-Lim phase update: momentum extrapolation, then renormalise to
        # unit magnitude so only the phase is kept (eps guards against divide-by-zero).
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + eps

    # Final inverse MCLT using the converged phase estimate.
    return mdct.fast.imclt(S * angles,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,
                        padding=0)
