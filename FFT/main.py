from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import concurrent.futures

def STE(frame):
    """
    Trả về giá trị STE một frame.
            Parameters:
                    frame (ndarray): Một frame của tín hiệu
            Returns:
                    (float): Giá trị STE của frame
    """
    return np.sum(frame**2)

def calculate(signal, Fs, frame_shift_amount_in_sample, frame_length_in_sample):
    """
    Tính giá trị STE của các frame trong tín hiệu và tìm các biên Voice/Unvoice.
            Parameters:
                    signal (ndarray): Array chứa tín hiệu
                    Fs (int): Tần số lấy mẫu của tín hiệu
                    frame_shift_amount_in_sample (int): Độ dịch khung với đơn vị mẫu
                    frame_length_in_sample (int): Độ dài khung với đơn vị mẫu
            Returns:
                    t_STE (ndarray): mảng chứa thời gian theo đơn vị giây ứng với thời gian bắt đầu mỗi frame
                    STE_of_signal (ndarray): mảng chứa các giá trị STE của mỗi frame của tín hiệu
                    program_borderlines (ndarray): mảng chứa các biên thời gian phân biệt Voice/Unvoice của tín hiệu
    """
    STE_of_signal, t_STE, program_borderlines = np.zeros(0), np.zeros(0), np.zeros(0)
    for i in range(0, len(signal), frame_shift_amount_in_sample):
        frame = signal[i:i + frame_length_in_sample + 1]
        ste = STE(frame)
        STE_of_signal = np.append(STE_of_signal, [ste])
        t_STE = np.append(t_STE, [i])
    STE_of_signal /= STE_of_signal.max()
    t_STE /= Fs
    VU = np.array([1 if STE_of_signal[i] >= STE_Threhold else 0 for i in range(len(STE_of_signal))])
    program_borderlines = np.array([t_STE[i] for i in range(1, len(VU)) if VU[i] != VU[i - 1]])
    return t_STE, STE_of_signal, program_borderlines

def calculateThread(signal_name):
    """
    Tìm mảng giá trị STE của tín hiệu và biên Voice/Unvoice rồi lưu lại dữ liệu.
            Parameters:
                    signal_name (str): tên tín hiệu
            Returns: None
    """
    print(f'Calculating {signal_name}')
    Fs, signal = wavfile.read(f'{path}/{signal_name}.wav')
    signal = signal / np.max(signal)
    length_signal_in_seconds = len(signal) / Fs
    frame_shift_amount_in_sample = int(frame_shift_amount_in_second * Fs)
    frame_length_in_sample = int(frame_length_in_second * Fs)
    with open(f'{path}/{signal_name}.lab', 'r') as file_lab:
        lines = file_lab.readlines()
        lab_borderlines = [float(j) for i in lines[:-2] for j in i.split()[:2] if i.split()[-1] != 'sil']
        F0mean = float(lines[-2].strip().split()[-1])
        F0std = float(lines[-1].strip().split()[-1])
    t_STE, STE_of_signal, program_borderlines = calculate(signal, Fs, frame_shift_amount_in_sample, frame_length_in_sample)
    datas.append((signal, t_STE, STE_of_signal, Fs, length_signal_in_seconds, signal_name, lab_borderlines, program_borderlines, F0mean, F0std))
    print(f'Calculate {signal_name} done !')

def hamming(frame_length_in_sample):
    """
    Trả về cửa sổ hamming với độ dài tương ứng.
            Parameters:
                    frame_length_in_sample (int): Độ dài cửa sổ hamming với đơn vị mẫu
            Returns:
                    (ndarray): cửa sổ hamming với độ dài tương ứng
    """
    return np.array(
        [(.54 - .46 * np.cos(2 * np.pi * i / (frame_length_in_sample - 1))) for i in range(frame_length_in_sample)])

def findHarmonics(spectrum):
    """
    Tìm dãy các đỉnh cách đều nhau có thể là hài của phổ tín hiệu.
            Parameters:
                    spectrum (ndarray): Array chứa phổ của tín hiệu
            Returns:
                    (list): mảng chứa các dãy các đỉnh cách đều nhau mà có thể là hài của phổ tín hiệu
    """
    # Tìm các đỉnh của phổ trong khoảng từ 0 - 2000Hz
    peaks = find_peaks(spectrum[:200])[0]
    possible_harmonics = []
    # Xét từng đỉnh tìm được
    for i in range(peaks.size):
        l = []
        # Nếu đỉnh này trong khoảng 70 - 400Hz
        if 7 <= peaks[i] <= 40:
            l.append(peaks[i])
            j = i + 1
            index = i
            # Tìm tất cả các đỉnh khác cách đều đỉnh đó một khoảng cách bằng khoảng cách đỉnh đó đến 0 với một sai số
            # là max_f_shift
            while j < peaks.size:
                f = peaks[j] - peaks[index]
                if f > 2 * peaks[i]:
                    break
                if abs(f - peaks[i]) <= max_f_shift:
                    index = j
                    l.append(peaks[j])
                j += 1
            if len(l) >= min_consecutive:
                possible_harmonics.append(l)
    possible_harmonics.sort(key=lambda x: -len(x))
    return possible_harmonics


def calF(harmonics):
    """
    Tìm khoảng cách trung bình giữa các đỉnh được đưa vào.
            Parameters:
                    harmonics (ndarray): Array chứa dãy các đỉnh cách đều nhau mà có thể là hài
            Returns:
                    (float): khoảng cách trung bình giữa các đỉnh
    """
    harmonics = [0] + harmonics
    rs = 0
    for i in range(1, len(harmonics)):
        rs += harmonics[i] - harmonics[i - 1]
    return rs / (len(harmonics) - 1) * 10


def findF0(l, possible_harmonics):
    """
    Trả về dãy các đỉnh có khả năng là hài nhất và tính F0 từ dãy đó.
            Parameters:
                    l (list): Array chứa tín hiệu
                    possible_harmonics (list): Tần số lấy mẫu của tín hiệu
            Returns:
                    (float): tần số F0 tìm được nếu có, nếu không có thì trả về -1
                    (list): dãy các đỉnh có khả năng là hài nhất, nếu không có thì trả về []
    """
    # Tìm khoảng cách trung bình từ mỗi dãy đỉnh trong possible_harmonics
    F0s = [calF(harmonics) for harmonics in possible_harmonics]
    # Nếu không có các giá trị F0 đã tính trước đó để so sánh thì trả về giá trị f đầu tiên trong mảng F0s, là giá trị
    # f có khả năng là F0 cao nhất.
    if len(l) < number_of_F0_compared:
        return F0s[0], possible_harmonics[0]
    l_mean = np.mean(l)
    for f in F0s:
        # Nếu độ lệch tần số của f so với trung bình các giá trị F0 đã tính trước đó trong khoảng cho phép thì lấy giá
        # trị tần số f đó làm F0.
        if abs(f - l_mean) <= max_f_shift * 10:
            return f, possible_harmonics[F0s.index(f)]
    return -1, []


def spectrum():
    """ Duyệt qua từng tín hiệu, lấy ra dữ liệu đã tính toán trước đó để thực hiện tính phổ của từng frame và tìm giá trị F0 """
    # Lấy ra dữ liệu đã tính trước đó của từng tín hiệu
    for d in range(len(datas)):
        # Khởi tạo giá trị
        data = datas[d]
        signal, program_borderline, Fs = data[0], data[7], data[3]
        frame_length_in_sample = int(frame_length_in_second * Fs)
        window = hamming(int(frame_length_in_second * Fs))
        F0, F0_times = [], []
        voice_f = -1
        harmonics, voice_log_mag_spectrum, unvoice_log_mag_spectrum = [], [], np.zeros(0)
        # Duyệt qua từng đoạn biên thời gian
        for i in range(0, len(program_borderline), 2):
            start, end = int(program_borderline[i] * Fs), int(program_borderline[i + 1] * Fs)
            for j in range(start, end, int(frame_shift_amount_in_second * Fs)):
                if len(signal[j:j + frame_length_in_sample]) != frame_length_in_sample:
                    break
                # Tính phổ biên độ và phổ log
                amplitude_spectrum = fft(signal[j:j + frame_length_in_sample] * window, n_fft)
                log_mag_spectrum = 10 * np.log10(abs(amplitude_spectrum))
                # Tìm dãy các đỉnh có khả năng là hài
                possible_harmonics = findHarmonics(log_mag_spectrum)
                if possible_harmonics:
                    f, _ = findF0(F0[-number_of_F0_compared:], possible_harmonics)
                    # Nếu tìm thấy F0 thì lưu lại giá trị F0 và mốc thời gian của frame
                    if f != -1:
                        # if i == 8:
                        #     print(f, _)
                        #     plt.figure('spectrum')
                        #     plt.plot(np.arange(len(log_mag_spectrum)) * 10, log_mag_spectrum)
                        #     plt.xlim((0, 2000))
                        #     plt.show()
                        # Lưu lại dữ liệu của một frame voice dùng cho việc plot
                        if voice_f == -1 and len(F0[-10:]) == 10:
                            voice_f = f
                            harmonics = _
                            voice_log_mag_spectrum = log_mag_spectrum
                        F0.append(f)
                        F0_times.append(j / Fs)
                # Lưu lại dữ liệu một frame unvoice dùng cho việc plot
                elif not unvoice_log_mag_spectrum.size:
                    unvoice_log_mag_spectrum = log_mag_spectrum
        # Kiểm tra nếu chưa có dữ liệu frame unvoice nào thì lấy dữ liệu frame đầu tiên của tín hiệu
        if not unvoice_log_mag_spectrum.size:
            amplitude_spectrum = fft(signal[:frame_length_in_sample] * window, n_fft)
            unvoice_log_mag_spectrum = 10 * np.log10(abs(amplitude_spectrum))

        F0 = medfil(F0)

        datas[d] += (F0, F0_times, voice_log_mag_spectrum, voice_f, harmonics, unvoice_log_mag_spectrum)

def medfil(l):
    l = np.concatenate(([0], l, [0]))
    indexes = []

    for i in range(1, l.size - 1):
        if abs(l[i] - l[i - 1]) <= pitch_shift and abs(l[i] - l[i + 1]) <= pitch_shift:
            indexes.append(i)

    l = l[indexes]

    for i in range(2, l.size - 2):
        l[i] = np.mean(l[i - 2:i + 2])

    return l

def genderDetect(name, F0mean):
    if 85 <= F0mean <= 155:
        return 'Male', 100 if name[2] == 'M' else 0
    if 155 < F0mean <= 255:
        return 'Female', 100 if name[2] == 'F' else 0
    return 'Unknown', 0

if __name__ == '__main__':
    # Khởi tạo các thông số mặc định
    frame_shift_amount_in_second, frame_length_in_second = .01, .024
    STE_Threhold = .0505
    max_f_shift = 4
    min_consecutive = 5
    pitch_shift = 10
    number_of_F0_compared = 4
    n_fft = 2**12
    datas = []
    path = '../TinHieuHuanLuyen-44k'
    list_of_signals = ['01MDA', '02FVA', '03MAB', '06FTB', '30FTN', '42FQT', '44MTT', '45MDV'] #
    # path = './TinHieuHuanLuyen'
    # list_of_signals = ['01MDA', '02FVA', '03MAB', '06FTB'] #
    # Duyệt qua từng tín hiệu và thực hiện tính toán
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        future = executor.map(calculateThread, list_of_signals)
        for i in future:
            if i:
                print(i)
    spectrum()

    print(f'{"File":^10}{"FFT (Hz)":^10}{"Lab (Hz)":^10}Relative Error (%) {"Gender":^10} Accuracy (%)')

    for data in datas:
        name, F0mean_fft, F0mean_lab = data[5], np.mean(data[10]), data[8]
        gender, accuracy = genderDetect(name, F0mean_fft)

        print(f'{name:^10}{F0mean_fft:^10.1f}{F0mean_lab:^10.1f}{abs(F0mean_fft - F0mean_lab) / F0mean_lab * 100:^19.1f}{gender:^10}{accuracy:^13}')

