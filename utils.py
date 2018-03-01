import cv2
import numpy as np
from scipy import signal
import pandas as pd
import jade


# # # Methods # # #
def green(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg


def grd(roi1, roi2):  # Simple GRD method
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    roi1_red = roi1[:, :, 2]
    roi2_red = roi2[:, :, 2]
    avg_green = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    avg_red = (np.mean(roi1_red) + np.mean(roi2_red)) / 2.0
    avg = avg_red - avg_green
    return avg


def return_avg(roi1, roi2):
    roi1_blue = roi1[:, :, 0]
    roi2_blue = roi2[:, :, 0]
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    roi1_red = roi1[:, :, 2]
    roi2_red = roi2[:, :, 2]
    avg_blue = (np.mean(roi1_blue) + np.mean(roi2_blue)) / 2.0
    avg_green = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    avg_red = (np.mean(roi1_red) + np.mean(roi2_red)) / 2.0
    return avg_red, avg_green, avg_blue


def adaptive_grd(r, g, b, fps):
    # Convert them to array
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    # Filter each channel with bandpass filter
    filtered_r = butterworth_filter(r, 0.8, 3.4, fps, order=5)
    filtered_g = butterworth_filter(g, 0.8, 3.4, fps, order=5)

    # Get the normalized factor and alpha-beta factors
    normalized_factor = np.sqrt(r**2 + g**2 + b**2)
    ab_r = r / normalized_factor
    ab_g = g / normalized_factor

    # Calculate the result
    sig = filtered_g / ab_g - filtered_r / ab_r
    return sig


def jade_ica_process(rgb, fs):
    B = jade.jadeR(rgb)
    # Y = B * matrix(rgb)
    A = np.dot(B, rgb)
    A = np.asarray(A)
    '''
    ica_result = max(abs(np.correlate(A[0], rgb[1])[0]),
                     abs(np.correlate(A[1], rgb[1])[0]),
                     abs(np.correlate(A[2], rgb[1])[0]))

    if abs(np.correlate(A[0], rgb[1])[0]) == ica_result:
        result = A[0]
    elif abs(np.correlate(A[1], rgb[1])[0]) == ica_result:
        result = A[1]
    elif abs(np.correlate(A[2], rgb[1])[0]) == ica_result:
        result = A[2]
    else:
        raise Exception("Invalid ICA!")
    '''
    f, pxx_den0 = signal.periodogram(A[0], fs)
    f, pxx_den1 = signal.periodogram(A[1], fs)
    f, pxx_den2 = signal.periodogram(A[2], fs)
    pxx_den0 = max(pxx_den0)
    pxx_den1 = max(pxx_den1)
    pxx_den2 = max(pxx_den2)
    ica_result = max(pxx_den0, pxx_den1, pxx_den2)
    if pxx_den0 == ica_result:
        result = A[0]
    elif pxx_den1 == ica_result:
        result = A[1]
    elif pxx_den2 == ica_result:
        result = A[2]
    else:
        raise Exception("Invalid ICA!")
    return result


def sb_pos(rgb, total_frame):
    # Parameters of window for POS algorithm (in 20 fps).
    # (1) L = 32 (1.6 s), B = [3,6];
    # (2) L = 64 (3.2 s), B = [4,12];
    # (3) L = 128 (6.4 s), B = [6,24];
    # (4) L = 256 (12.8 s), B = [10,50];
    # (5) L = 512 (25.6 s), B = [18,100].
    # Here the parameter group (1) and (2) are performed well.
    pos = np.array([[0, 1, -1], [-2, 1, 1]])
    l = 64
    b = [4, 12]
    p = np.zeros([1, total_frame])
    for t in range(int(total_frame-l+1)):
        c = rgb[:, t:t+l-1]
        normalized_c = np.dot(np.linalg.inv(np.diag(c.mean(axis=1))), c)-1
        f = np.fft.fft(normalized_c)
        s = np.dot(pos, f)
        z = s[0, :]+abs(s[0, :])/abs(s[1, :])*s[1, :]
        z_avg = z*abs(z/(np.sum(f, axis=0)))
        z_avg[:b[0]] = 0
        z_avg[b[1]+1:-1] = 0
        p_avg = np.fft.ifft(z_avg)
        p_avg = p_avg.real
        p[0, t:t+l-1] = p[0, t:t+l-1] + (p_avg-np.mean(p_avg))/np.std(p_avg)
    return p[0]


# # # Processing # # #
# Creates the specified Butterworth filter and applies it.
# See:  http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butterworth_filter(data, low, high, sampling_rate, order=5):
    nyq = sampling_rate * 0.5
    low /= nyq
    high /= nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


# Design a FIR filter with 128-point hamming window and applies it.
# FIXME: It doesn't work on the buffer whose length is 300. Maybe need to padlen or decrease ntaps.
def fir_filter(data, low, high, sampling_rate, ntaps=128):
    nyq = 0.5 * sampling_rate
    b = signal.firwin(ntaps, [low, high], nyq=nyq, pass_zero=False, window='hamming', scale=False)
    return signal.filtfilt(b, 1, data)  # signal.lfilter(b, [1.0], data)


def normalize(sig):
    return (sig - np.mean(sig)) / np.std(sig)


def sliding_window_demean(data, num_windows):
    window_size = int(round(len(data) / num_windows))
    demeaned = np.zeros(data.shape)
    for i in range(0, len(data), window_size):
        if i + window_size > len(data):
            window_size = len(data) - i
        temp_slice = data[i:i + window_size]
        if temp_slice.size == 0:
            print('Empty Slice: size={0}, i={1}, window_size={2}'.format(data.size, i, window_size))
            print(temp_slice)
        demeaned[i:i + window_size] = temp_slice - np.mean(temp_slice)
    return demeaned


def detrending(data):
    try:
        detrended = signal.detrend(np.array(data), type='linear')
    except ValueError:
        print("Captured invalid data and they have been already rectified.")
        temp_values = pd.DataFrame(data)
        temp_values = temp_values.fillna(temp_values.mean())
        values = np.ndarray.flatten(np.array(temp_values))
        data = values.tolist()
        detrended = signal.detrend(np.array(data), type='linear')
    return detrended, data


# # # Gets ROIs # # #
# Gets the region of interest for the forehead.
def get_forehead_roi(face_points):
    # Store the points in a Numpy array so we can easily get the min and max for x and y via slicing
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Forehead area between eyebrows
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)


# Gets the region of interest for the nose.
def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Nose and cheeks
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)


# Gets region of interest that includes forehead, eyes, and nose.
# Note:  Combination of forehead and nose performs better.
#        This is probably because this ROI includes the eyes, and eye blinking adds noise.
def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Only keep the points that correspond to the internal features of the face (e.g. mouth, nose, eyes, brows).
    # The points outlining the jaw are discarded.
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1]))

    center_x = min_x + (max_x - min_x) / 2
    # center_y = min_y + (max_y - min_y) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return int(left), int(right), int(top), int(bottom)


# # # GUI # # #
# Draws the heart rate graph in the GUI window.
def draw_graph(data, graph_width, graph_height, buffer_max_size=300):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / buffer_max_size
    scale_factor_y = 30
    midpoint_y = graph_height / 2
    for i in range(0, data.shape[0] - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + data[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + data[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph


# Draws the heart rate text (BPM) in the GUI window.
def draw_bpm(bpm_str, bpm_width, bpm_height):
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    # Draw gray line to separate graph from BPM display
    bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height / 2 + bpm_text_base)
    cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2.7, color=(0, 255, 0), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
    bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
    return bpm_display


# Draws the current frames per second in the GUI window.
# This can be turned off by setting the "show_fps" constant to False.
def draw_fps(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
    cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 255, 0))
    return frame
