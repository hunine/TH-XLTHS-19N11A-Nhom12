import concurrent.futures
from scipy.io import wavfile
import numpy as np

def find_dips(l):
    return np.array([i for i in range(1, l.size - 1) if l[i - 1] > l[i] and l[i + 1] > l[i] and l[i] <= amdf_threshold]) if l.size > 2 else np.zeros(0)

def amdf(frame, max_period_in_sample, min_period_in_sample):
    d = np.zeros(max_period_in_sample + 1)
    if frame.size >= max_period_in_sample:
        for n in range(min_period_in_sample, max_period_in_sample + 1):
            d[n] = np.sum(abs(frame[:frame.size - n] - frame[n:]) / (frame.size - n))
    else:
        return -1

    d /= np.max(d)

    dips = find_dips(d)

    if dips.size == 0:
        return -1

    return d[dips[np.argmin(d[dips])]]

def calculateThread(signal_name):
    print(f'Calculating {signal_name}')

    Fs, signal = wavfile.read(f'{path}{signal_name}.wav')
    signal = signal / max(np.max(signal), abs(np.min(signal)))
    dips_voice, dips_silent = np.zeros(0), np.zeros(0)
    frame_length_in_sample = int(frame_length_in_seconds * Fs)
    max_period_in_sample, min_period_in_sample = Fs // min_pitch_value, Fs // max_pitch_value

    with open(f'{path}{signal_name}.lab', 'r') as file_lab:
        lines = file_lab.readlines()[:-2]
        for line in lines:
            start, end, label = line.split()

            for i in range(int(float(start) * Fs), int(float(end) * Fs), frame_length_in_sample):
                dip = amdf(signal[i:i + frame_length_in_sample], max_period_in_sample, min_period_in_sample)

                if dip != -1:
                    if label == 'sil':
                        dips_silent = np.append(dips_silent, [dip])
                    else:
                        dips_voice = np.append(dips_voice, [dip])

    datas.append((signal_name, np.mean(dips_voice), np.std(dips_voice), np.mean(dips_silent), np.std(dips_silent)))

    print(f'Calculate {signal_name} done !')

if __name__ == '__main__':
    list_of_signals = ['01MDA', '02FVA', '03MAB', '06FTB', '30FTN', '42FQT', '44MTT', '45MDV'] #
    path = '../TinHieuHuanLuyen-44k/'
    amdf_threshold = 1
    frame_length_in_seconds = 0.025
    max_pitch_value = 450
    min_pitch_value = 70
    datas = []

    with concurrent.futures.ThreadPoolExecutor(20) as executor:
        future = executor.map(calculateThread, list_of_signals)
        for i in future:
            if i:
                print(i)
    print(f'{"File":10}{"MeanVoice":^10}{"StdVoice":^10}{"MeanSilent":^12}{"StdSilent":^12}(meanVoice + stdVoice + meanSilent - stdSilent) / 2')
    threshold = np.zeros(0)
    for name, meanVoice, stdVoice, meanSilent, stdSilent in datas:
        threshold = np.append(threshold, [(meanVoice + stdVoice + meanSilent - stdSilent) / 2])
        print(f'{name:10}{meanVoice:^10.3f}{stdVoice:^10.3f}{meanSilent:^12.3f}{stdSilent:^12.3f}{threshold[-1]:^51.3f}')
    print(f'Threshold = {np.mean(threshold)}')