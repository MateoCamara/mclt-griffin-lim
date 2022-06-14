import warnings

import librosa
import librosa.display
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

    Given a Modulated Complex Lapped Transform magnitude matrix (``S``), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-MCLT
    operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    .. [#] H. Malvar,
       “A modulated complex lapped transform and its applications to audio processing,”
        presented at the 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Proceedings. ICASSP99, 1999.

    Parameters
    ----------
    S : np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
        An array of MCLT magnitudes as produced by `mdct.mclt`.

    n_iter : int > 0
        The number of iterations to run

    frame_length : None or int > 0
        The frame length of the MCLT.  If not provided, it will default to ``2 * (S.shape[-2] - 1)``.
        It needs to be a power of 2.

    center : boolean
        If ``True``, the MCLT is assumed to use centered frames.
        If ``False``, the MCLT is assumed to use left-aligned frames.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``S`` is
        a magnitude spectrogram with no initial phase estimates.

        If `None`, then the phase is initialized from ``S``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` instance, the random number
        generator itself.

        If `None`, defaults to the current `np.random` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``S``

    See Also
    --------
    librosa
    mdct
    stft
    magphase

    """

    rng = np.random
    if isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

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

    # Infer frame_length from the spectrogram shape
    if frame_length is None:
        frame_length = 2 * (S.shape[-2] - 1)
        warnings.warn(
            "Mclt requires frame_length to be a power of 2, it's better to introduce frame_length manually"
        )

    if not ((frame_length & (frame_length-1) == 0) and frame_length != 0):
        raise ParameterError(
            "frame_length is not a power of 2"
        )

    hop_length = int(frame_length / 2)

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(S.shape, dtype=np.complex64)
    eps = librosa.util.tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.0

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases

        inverse = mdct.fast.imclt(S * angles,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,
                        padding=0,
                        outlength=int(sample_rate * frame_length / 2))

        # Rebuild the spectrogram
        rebuilt = mdct.fast.mclt(inverse,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,
                        padding=0)[:, :-1]


        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + eps

    # Return the final phase estimates
    return mdct.fast.imclt(S * angles,
                        framelength=frame_length,
                        hopsize=hop_length,
                        overlap=2,
                        centered=center,
                        window=stft.stft.cosine,
                        padding=0,
                        outlength=int(sample_rate * frame_length / 2))