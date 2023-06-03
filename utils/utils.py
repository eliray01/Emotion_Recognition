import numpy as np
import scipy
import neurokit2 as nk

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

def rr_interval(ecg_cleaned):
    
    """This function takes ECG and returns RR intervals"""
    
    rpeaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=250, correct_artifacts=True)
    time_diff = []
    for i in range(len(info['ECG_R_Peaks'])):
        if i == len(info['ECG_R_Peaks'])-1:
            break
        else:
            time_diff.append(info['ECG_R_Peaks'][i+1] - info['ECG_R_Peaks'][i])
    return time_diff, info

def exctract_hrv(ecg_cleaned, baseline_mean, baseline_std, coefficient = 1.5):
    
    up = 0
    down = 0
    calm = 0
    
    increase_idx = []
    decrease_idx = []
    calm_idx = []
    rr_intervals, info = rr_interval(ecg_cleaned)
    
    for k in range(len(rr_intervals)):
        if rr_intervals[k] < baseline_mean-coefficient*baseline_std:
            down+=1
            decrease_idx.append(k)
        elif rr_intervals[k] > baseline_mean+coefficient*baseline_std:
            up+=1
            increase_idx.append(k)
        else:
            calm+=1
            calm_idx.append(k)
    if calm != 0 and up == 0 and down == 0:
        #print('calm', calm_idx)
        return 'calm', calm_idx
    elif calm == 0 and up == 0 and down == 0:
        #print('calm', calm_idx)
        return 'calm', calm_idx
    elif down >= up:
        #print('decrease', decrease_idx)
        return 'decrease', decrease_idx
    elif up > down:
        #print('increase', increase_idx)
        return 'increase', increase_idx

def exctract_hr(signals, info, indexes_for_hr):
    
    #signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=250, method = 'neurokit') 
    hr_derivative = np.gradient(signals['ECG_Rate'])
    hr_increases = np.where(hr_derivative > 0)[0] #Compute indexes where HR increasing 
    hr_decreases = np.where(hr_derivative < 0)[0] #and decreasing
    time = np.arange(len(signals['ECG_Rate']))
    
    increase = 0
    decrease = 0
    calm = 0
    for idx in indexes_for_hr:
        start = info['ECG_R_Peaks'][idx]
        end = info['ECG_R_Peaks'][idx+1]

        moment = np.arange(start,end+1)
    
        increase_count = np.sum(np.isin(moment, time[hr_increases]))
        decrease_count = np.sum(np.isin(moment, time[hr_decreases]))
        
        if increase_count > decrease_count:
            #print(' HR increase')
            increase+=1
        elif decrease_count > increase_count:
            #print(' HR decrease')
            decrease+=1
        else:
            #print(' HR calm')
            calm +=1
    max_value = max(increase, decrease, calm)
    if max_value == increase:
        #print('increase')
        return 'increase'
    elif max_value == decrease:
        #print('decrease')
        return 'decrease'
    else:
        #print('calm')
        return 'calm'
        
    

def extract_p_wave(signals):
    p_wave_amplitude = []
    offsets = signals[signals['ECG_P_Offsets'] == 1]
    onsets = signals[signals['ECG_P_Onsets'] == 1]
    peaks = signals[signals['ECG_P_Peaks'] == 1]
    peaks_len = len(signals[signals['ECG_P_Peaks'] == 1])
    offsets_len = len(signals[signals['ECG_P_Offsets'] == 1])
    onsets_len = len(signals[signals['ECG_P_Onsets'] == 1])
    
    if peaks_len != offsets_len or peaks_len != onsets_len or offsets_len != onsets_len:
        if min(peaks_len,offsets_len,onsets_len) == peaks_len:
            offsets = offsets.drop(index=offsets.index[0])
            onsets = onsets.drop(index=onsets.index[0])
        elif min(peaks_len,offsets_len,onsets_len) == offsets_len:
            peaks = peaks.drop(index=peaks.index[0])
            onsets = onsets.drop(index=onsets.index[0])
        elif min(peaks_len,offsets_len,onsets_len) == onsets_len:
            offsets = offsets.drop(index=offsets.index[0])
            peaks = peaks.drop(index=peaks.index[0])
        for i in range(len(peaks)):
             p_wave_amplitude.append(offsets.index[i]-onsets.index[i])
        return np.mean(p_wave_amplitude)
    else:
        for i in range(len(peaks)):
             p_wave_amplitude.append(offsets.index[i]-onsets.index[i])
        return np.mean(p_wave_amplitude)
    
def moving_average(data, window_size):
    """Auxiliary function for extract_sgr"""
    # Create an empty array to hold the moving averages
    ma = np.empty_like(data)
    ma.fill(np.nan)
    
    # Compute the moving average for each window
    for i in range(len(data)):
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        window_data = data[window_start:window_end]
        ma[i] = np.mean(window_data)
        
    return ma

def extract_sgr(sgr, info, indexes_for_sgr):
    if len(sgr) > 1250:
        sgr_data_filtered = moving_average(sgr,500) # Filtering for baselines 
    else:
        sgr_data_filtered = moving_average(sgr,25) # Filtering for blocks
        
    sgr_derivative = np.gradient(sgr_data_filtered)
    
    sgr_increases = np.where(sgr_derivative > 0)[0]
    sgr_decreases = np.where(sgr_derivative < 0)[0]
    time = np.arange(len(sgr_data_filtered))
    
    increase = 0
    decrease = 0
    calm = 0
    for idx in indexes_for_sgr:
        start = info['ECG_R_Peaks'][idx]
        end = info['ECG_R_Peaks'][idx+1]

        moment = np.arange(start,end+1)
    
        increase_count = np.sum(np.isin(moment, time[sgr_increases]))
        decrease_count = np.sum(np.isin(moment, time[sgr_decreases]))
        if increase_count > decrease_count:
            #print(' SRG increase')
            increase+=1
        elif decrease_count > increase_count:
            #print(' SRG decrease')
            decrease+=1
        else:
            #print('SRG calm')
            calm +=1
    max_value = max(increase, decrease, calm)
    if max_value == increase:
        #print('increase')
        return "increase"
    elif max_value == decrease:
        #print('decrease')
        return "decrease"
    else:
        #print('calm')
        return "calm"
    
def ppg_amplitude(peaks,signal,baseline_mean_amp=None,baseline_std_amp=None):
    amplitude =[]
    for up, down in zip(signal['PPG_Clean'][peaks[0][0]], signal['PPG_Clean'][peaks[1][0]]):
        amplitude.append(up-down)
        
    if baseline_mean_amp is None or baseline_std_amp is None:
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        return mean_amp, std_amp
    
    up = 0
    down = 0
    calm = 0

    for k in range(len(amplitude)):
        if amplitude[k] < baseline_mean_amp-baseline_std_amp:
            down+=1
        elif amplitude[k] > baseline_mean_amp+baseline_std_amp:
            up+=1
        else:
            calm+=1
    if calm != 0 and up == 0 and down == 0:
        #print('calm')
        return 'calm', amplitude
    elif calm == 0 and up == 0 and down == 0:
        #print('calm')
        return 'calm', amplitude
    elif down >= up:
        #print('decrease')
        return 'decrease', amplitude
    elif up > down:
        #print('increase')
        return 'increase', amplitude
    
def ppg_frequency(info_ppg, baseline_mean_freq=None, baseline_std_freq=None):
    
    time_diff_ppg = []
    for i in range(len(info_ppg['PPG_Peaks'])):
        if i == len(info_ppg['PPG_Peaks'])-1:
            break
        else:
            time_diff_ppg.append(info_ppg['PPG_Peaks'][i+1] - info_ppg['PPG_Peaks'][i])
            
    if baseline_mean_freq is None or baseline_std_freq is None:
        mean_freq = np.mean(time_diff_ppg)
        std_freq = np.std(time_diff_ppg)
        return mean_freq, std_freq
    
    up = 0
    down = 0
    calm = 0
    
    for k in range(len(time_diff_ppg)):
        if time_diff_ppg[k] < baseline_mean_freq-baseline_std_freq:
            down+=1
        elif time_diff_ppg[k] > baseline_mean_freq+baseline_std_freq:
            up+=1
        else:
            calm+=1
            
    if calm != 0 and up == 0 and down == 0:
        #print('calm')
        return 'calm', time_diff_ppg
    elif calm == 0 and up == 0 and down == 0:
        #print('calm')
        return 'calm', time_diff_ppg
    elif down >= up:
        #print('decrease')
        return 'decrease', time_diff_ppg
    elif up > down:
        #print('increase')
        return 'increase', time_diff_ppg

def lfhf_ratio(ecg_cleaned, baseline_lf_hf = None):
    if baseline_lf_hf == None:
        time_diff, info = rr_interval(ecg_cleaned = ecg_cleaned)
        lfhf_ratio = get_frequency_domain_features(time_diff, method = 'welch', sampling_frequency = 2.5)['lf_hf_ratio']
        return lfhf_ratio
    
    time_diff, info = rr_interval(ecg_cleaned = ecg_cleaned)
    lfhf_ratio = get_frequency_domain_features(time_diff, method = 'welch', sampling_frequency = 2.5)['lf_hf_ratio']
    if lfhf_ratio >= baseline_lf_hf:
        return 'increase'
    else:
        return 'decrease'
