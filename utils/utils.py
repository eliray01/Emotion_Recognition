import numpy as np
import scipy

def ppg_findpeaks_bishop(
    signal,
    show=False,
):
    """Implementation of Bishop SM, Ercole A (2018) Multi-scale peak and trough detection optimised
    for periodic and quasi-periodic neuroscience data. doi:10.1007/978-3-319-65798-1_39.

    Currently designed for short signals of relatively low sampling frequencies (e.g. 6 seconds at
    100 Hz). Also, the function currently only returns peaks, but it does identify pulse onsets too.
    """

    # TODO: create ppg_peaks() that also returns onsets and stuff
    # Setup
    N = len(signal)
    L = int(np.ceil(N / 2) - 1)

    # Step 1: calculate local maxima and local minima scalograms

    # - detrend: this removes the best-fit straight line
    x = scipy.signal.detrend(signal, type="linear")

    # - initialise LMS matrices
    m_max = np.full((L, N), False)
    m_min = np.full((L, N), False)

    # - populate LMS matrices
    for k in range(1, L):  # scalogram scales
        for i in range(k + 2, N - k + 1):
            if x[i - 1] > x[i - k - 1] and x[i - 1] > x[i + k - 1]:
                m_max[k - 1, i - 1] = True
            if x[i - 1] < x[i - k - 1] and x[i - 1] < x[i + k - 1]:
                m_min[k - 1, i - 1] = True

    # Step 2: find the scale with the most local maxima (or local minima)
    # - row-wise summation (i.e. sum each row)
    gamma_max = np.sum(m_max, axis=1)
    # the "axis=1" option makes it row-wise
    gamma_min = np.sum(m_min, axis=1)
    # - find scale with the most local maxima (or local minima)
    lambda_max = np.argmax(gamma_max)
    lambda_min = np.argmax(gamma_min)

    # Step 3: Use lambda to remove all elements of m for which k>lambda
    m_max = m_max[: (lambda_max + 1), :]
    m_min = m_min[: (lambda_min + 1), :]

    # Step 4: Find peaks (and onsets)
    # - column-wise summation
    m_max_sum = np.sum(m_max == False, axis=0)
    m_min_sum = np.sum(m_min == False, axis=0)
    peaks = np.asarray(np.where(m_max_sum == 0)).astype(int)
    onsets = np.asarray(np.where(m_min_sum == 0)).astype(int)

    if show:
        _, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax0.plot(signal, label="signal")
        ax0.scatter(peaks, signal[peaks], c="r")
        ax0.scatter(onsets, signal[onsets], c="b")
        ax0.set_title("PPG Peaks (Method by Bishop et al., 2018)")
    return peaks, onsets