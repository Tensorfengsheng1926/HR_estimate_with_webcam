import dlib
import time
from utils import *


class Recorder:
    def __init__(self, detector, predictor, webcam, show_fps=True):
        # Initialize
        self.detector = detector
        self.predictor = predictor
        self.webcam = webcam
        self.show_fps = show_fps  # Controls whether the FPS is displayed in top-left of GUI window.

        # Constants
        self.window_name = 'Esc to exit'
        self.buffer_max_size = 300
        self.min_hz = 0.83  # 50 BPM
        self.max_hz = 3.33  # 200 BPM
        self.graph_height = 200
        self.graph_width = 0
        self.bpm_display_width = 0
        self.min_frames = 200  # The pulse rate will be shown after processed 'min_frames' frames.
        self.bpm = 0
        self.last_bpm = 0

        # Lists for storing ROIs
        self.fh_roi = []
        self.nose_roi = []

        # Lists for storing video frame data
        self.values = []
        self.B = []
        self.G = []
        self.R = []
        self.times = []

    def detected_gui(self, curr_buffer_size):
        # If there's not enough data to compute HR, show an empty graph with loading text and
        # the BPM placeholder
        graph = np.zeros((self.graph_height, self.graph_width, 3), np.uint8)
        pct = int(round(float(curr_buffer_size) / self.min_frames * 100.0))
        loading_text = 'Computing pulse: ' + str(pct) + '%'
        loading_size, loading_base = cv2.getTextSize(loading_text, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                     fontScale=1, thickness=1)
        loading_x = int((self.graph_width - loading_size[0]) / 2)
        loading_y = int(self.graph_height / 2 + loading_base)
        cv2.putText(graph, loading_text, (loading_x, loading_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(0, 255, 0), thickness=1)

        waiting_text = 'Please wait until the value to become stable.'
        cv2.putText(graph, waiting_text, (loading_x - 25, loading_y + 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.6, color=(0, 255, 0), thickness=1)
        bpm_display = draw_bpm('--', self.bpm_display_width, self.graph_height)
        return graph, bpm_display

    def no_face_gui(self, view):
        self.graph_width = int(view.shape[1] * 0.75)
        self.bpm_display_width = view.shape[1] - self.graph_width
        graph = np.zeros((self.graph_height, self.graph_width, 3), np.uint8)
        loading_text = 'No face or more than one face detected.'
        loading_size, loading_base = cv2.getTextSize(loading_text, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                     fontScale=1, thickness=1)
        loading_x = int((self.graph_width - loading_size[0] / 1.5) / 2)
        loading_y = int(self.graph_height / 2 + loading_base)
        cv2.putText(graph, loading_text, (loading_x, loading_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.7, color=(0, 255, 0), thickness=1)

        bpm_display = draw_bpm('--', self.bpm_display_width, self.graph_height)
        graph = np.hstack((graph, bpm_display))
        view = np.vstack((view, graph))
        return view

    # Main functions.
    def run_pulse_observer_in_g_method(self):
        cv2.namedWindow(self.window_name)

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(self.window_name, 0) == 0:
            r, frame = self.webcam.read()

            # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            # The original frame will be used to compute heart rate.
            view = np.array(frame)

            # Detect face using dlib
            faces = self.detector(frame, 0)
            if len(faces) == 1:

                # Heart rate graph gets 75% of window width.  BPM gets 25%.
                self.graph_width = int(view.shape[1] * 0.75)
                self.bpm_display_width = view.shape[1] - self.graph_width
                face_points = self.predictor(frame, faces[0])

                # Get the regions of interest.
                fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
                nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                # Slice out the regions of interest (ROI)
                self.fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                self.nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]

                # Average values and add to list
                avg = green(self.fh_roi, self.nose_roi)
                self.values.append(avg)

                # Add time to list
                self.times.append(time.time())

                # Buffer is full, so pop the value off the top
                if len(self.times) > self.buffer_max_size:
                    self.values.pop(0)
                    self.times.pop(0)

                curr_buffer_size = len(self.times)

                # Don't try to compute pulse until we have at least the min. number of frames (e.g. 60)
                if curr_buffer_size > self.min_frames:
                    # Smooth the signal by detrending and demeaning
                    detrended, self.values = detrending(self.values)
                    demeaned = sliding_window_demean(detrended, 15)

                    # Compute relevant times
                    time_elapsed = self.times[-1] - self.times[0]
                    fps = curr_buffer_size / time_elapsed  # frames per second

                    # Filter signal with Butterworth bandpass filter
                    filtered = butterworth_filter(demeaned, self.min_hz, self.max_hz, fps, order=5)
                    filtered = check_nan(filtered)

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(filtered))

                    # Generate list of frequencies that correspond to the FFT values
                    freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                    # because they correspond to impossible BPM values.
                    while True:
                        max_idx = fft.argmax()
                        bps = freqs[max_idx]
                        if bps < self.min_hz or bps > self.max_hz:
                            print('BPM of {0} was discarded.'.format(bps * 60.0))
                            fft[max_idx] = 0
                        else:
                            bpm = bps * 60.0
                            break

                    # It's impossible for the heart rate to change more than 10% between samples,
                    # so use a weighted average to smooth the BPM with the last BPM.
                    if self.last_bpm > 0:
                        bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
                        self.last_bpm = bpm

                    graph = draw_graph(filtered, self.graph_width, self.graph_height, self.buffer_max_size)
                    bpm_display = draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.graph_height)

                    if self.show_fps:
                        view = draw_fps(view, fps)

                else:
                    graph, bpm_display = self.detected_gui(curr_buffer_size)

                # Show GUI in window
                graph = np.hstack((graph, bpm_display))
                view = np.vstack((view, graph))

            else:
                # No faces detected, so we must clear the lists of values and timestamps. Otherwise there will be a gap
                # in timestamps when a face is detected again.

                # GUI
                view = self.no_face_gui(view)

                # Clear values
                del self.values[:]
                del self.times[:]

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27 or r == 0:
                break

    def run_pulse_observer_in_grd_method(self):
        cv2.namedWindow(self.window_name)

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(self.window_name, 0) == 0:
            r, frame = self.webcam.read()

            # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            # The original frame will be used to compute heart rate.
            view = np.array(frame)

            # Detect face using dlib
            faces = self.detector(frame, 0)
            if len(faces) == 1:

                # Heart rate graph gets 75% of window width.  BPM gets 25%.
                self.graph_width = int(view.shape[1] * 0.75)
                self.bpm_display_width = view.shape[1] - self.graph_width
                face_points = self.predictor(frame, faces[0])

                # Get the regions of interest.
                fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
                nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                # Slice out the regions of interest (ROI)
                self.fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                self.nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]

                # Average values and add to list
                avg = grd(self.fh_roi, self.nose_roi)
                self.values.append(avg)

                # Add time to list
                self.times.append(time.time())

                # Buffer is full, so pop the value off the top
                if len(self.times) > self.buffer_max_size:
                    self.values.pop(0)
                    self.times.pop(0)

                curr_buffer_size = len(self.times)

                # Don't try to compute pulse until we have at least the min. number of frames (e.g. 60)
                if curr_buffer_size > self.min_frames:
                    # Smooth the signal by detrending and demeaning
                    detrended, self.values = detrending(self.values)
                    demeaned = sliding_window_demean(detrended, 15)

                    # Compute relevant times
                    time_elapsed = self.times[-1] - self.times[0]
                    fps = curr_buffer_size / time_elapsed  # frames per second

                    # Filter signal with Butterworth bandpass filter
                    filtered = butterworth_filter(demeaned, self.min_hz, self.max_hz, fps, order=5)
                    filtered = check_nan(filtered)

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(filtered))

                    # Generate list of frequencies that correspond to the FFT values
                    freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                    # because they correspond to impossible BPM values.
                    while True:
                        max_idx = fft.argmax()
                        bps = freqs[max_idx]
                        if bps < self.min_hz or bps > self.max_hz:
                            print('BPM of {0} was discarded.'.format(bps * 60.0))
                            fft[max_idx] = 0
                        else:
                            bpm = bps * 60.0
                            break

                    # It's impossible for the heart rate to change more than 10% between samples,
                    # so use a weighted average to smooth the BPM with the last BPM.
                    if self.last_bpm > 0:
                        bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
                        self.last_bpm = bpm

                    graph = draw_graph(filtered, self.graph_width, self.graph_height, self.buffer_max_size)
                    bpm_display = draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.graph_height)

                    if self.show_fps:
                        view = draw_fps(view, fps)

                else:
                    graph, bpm_display = self.detected_gui(curr_buffer_size)

                # Show GUI in window
                graph = np.hstack((graph, bpm_display))
                view = np.vstack((view, graph))

            else:
                # No faces detected, so we must clear the lists of values and timestamps. Otherwise there will be a gap
                # in timestamps when a face is detected again.

                # GUI
                view = self.no_face_gui(view)

                # Clear values
                del self.values[:]
                del self.times[:]

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27 or r == 0:
                break

    def run_pulse_observer_in_adaptive_grd_method(self):
        cv2.namedWindow(self.window_name)

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(self.window_name, 0) == 0:
            r, frame = self.webcam.read()

            # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            # The original frame will be used to compute heart rate.
            view = np.array(frame)

            # Detect face using dlib
            faces = self.detector(frame, 0)
            if len(faces) == 1:

                # Heart rate graph gets 75% of window width.  BPM gets 25%.
                self.graph_width = int(view.shape[1] * 0.75)
                self.bpm_display_width = view.shape[1] - self.graph_width
                face_points = self.predictor(frame, faces[0])

                # Get the regions of interest.
                fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
                nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                # Slice out the regions of interest (ROI)
                self.fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                self.nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]

                # Average values and add to list
                r, g, b = return_avg(self.fh_roi, self.nose_roi)
                self.R.append(r)
                self.G.append(g)
                self.B.append(b)

                # Add time to list
                self.times.append(time.time())

                # Buffer is full, so pop the value off the top
                if len(self.times) > self.buffer_max_size:
                    self.R.pop(0)
                    self.G.pop(0)
                    self.B.pop(0)
                    self.times.pop(0)

                curr_buffer_size = len(self.times)

                # Don't try to compute pulse until we have at least the min. number of frames (e.g. 60)
                if curr_buffer_size > self.min_frames:
                    # Compute relevant times
                    time_elapsed = self.times[-1] - self.times[0]
                    fps = curr_buffer_size / time_elapsed  # frames per second

                    # Filter signal with FIR
                    filtered = adaptive_grd(self.R, self.G, self.B, fps)
                    filtered = check_nan(filtered)

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(filtered))

                    # Generate list of frequencies that correspond to the FFT values
                    freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                    # because they correspond to impossible BPM values.
                    while True:
                        max_idx = fft.argmax()
                        bps = freqs[max_idx]
                        if bps < self.min_hz or bps > self.max_hz:
                            print('BPM of {0} was discarded.'.format(bps * 60.0))
                            fft[max_idx] = 0
                        else:
                            bpm = bps * 60.0
                            break

                    # It's impossible for the heart rate to change more than 10% between samples,
                    # so use a weighted average to smooth the BPM with the last BPM.
                    if self.last_bpm > 0:
                        bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
                        self.last_bpm = bpm

                    graph = draw_graph(filtered, self.graph_width, self.graph_height, self.buffer_max_size)
                    bpm_display = draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.graph_height)

                    if self.show_fps:
                        view = draw_fps(view, fps)

                else:
                    graph, bpm_display = self.detected_gui(curr_buffer_size)

                # Show GUI in window
                graph = np.hstack((graph, bpm_display))
                view = np.vstack((view, graph))

            else:
                # No faces detected, so we must clear the lists of values and timestamps. Otherwise there will be a gap
                # in timestamps when a face is detected again.

                # GUI
                view = self.no_face_gui(view)

                # Clear values
                del self.values[:]
                del self.R[:]
                del self.G[:]
                del self.B[:]
                del self.times[:]

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27 or r == 0:
                break

    def run_pulse_observer_in_ica_method(self):
        cv2.namedWindow(self.window_name)

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(self.window_name, 0) == 0:
            r, frame = self.webcam.read()

            # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            # The original frame will be used to compute heart rate.
            view = np.array(frame)

            # Detect face using dlib
            faces = self.detector(frame, 0)
            if len(faces) == 1:

                # Heart rate graph gets 75% of window width.  BPM gets 25%.
                self.graph_width = int(view.shape[1] * 0.75)
                self.bpm_display_width = view.shape[1] - self.graph_width
                face_points = self.predictor(frame, faces[0])

                # Get the regions of interest.
                fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
                nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                # Slice out the regions of interest (ROI)
                self.fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                self.nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]

                # Average values and add to list
                r, g, b = return_avg(self.fh_roi, self.nose_roi)
                self.R.append(r)
                self.G.append(g)
                self.B.append(b)

                # Add time to list
                self.times.append(time.time())

                # Buffer is full, so pop the value off the top
                if len(self.times) > self.buffer_max_size:
                    self.R.pop(0)
                    self.G.pop(0)
                    self.B.pop(0)
                    self.times.pop(0)

                curr_buffer_size = len(self.times)

                # Don't try to compute pulse until we have at least the min. number of frames (e.g. 60)
                if curr_buffer_size > self.min_frames:
                    # Smooth each channel by detrending and demeaning
                    detrended, self.R = detrending(self.R)
                    self.R = sliding_window_demean(detrended, 12).tolist()
                    detrended, self.G = detrending(self.G)
                    self.G = sliding_window_demean(detrended, 12).tolist()
                    detrended, self.B = detrending(self.B)
                    self.B = sliding_window_demean(detrended, 12).tolist()

                    # Compute relevant times
                    time_elapsed = self.times[-1] - self.times[0]
                    fps = curr_buffer_size / time_elapsed  # frames per second

                    # Concatenate three channels to matrix
                    rgb = np.array([self.R, self.G, self.B])

                    # BSS
                    filtered = jade_ica_process(rgb, fps)
                    filtered = check_nan(filtered)

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(filtered))

                    # Generate list of frequencies that correspond to the FFT values
                    freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                    # because they correspond to impossible BPM values.
                    while True:
                        max_idx = fft.argmax()
                        bps = freqs[max_idx]
                        if bps < self.min_hz or bps > self.max_hz:
                            print('BPM of {0} was discarded.'.format(bps * 60.0))
                            fft[max_idx] = 0
                        else:
                            bpm = bps * 60.0
                            break

                    # It's impossible for the heart rate to change more than 10% between samples,
                    # so use a weighted average to smooth the BPM with the last BPM.
                    if self.last_bpm > 0:
                        bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
                        self.last_bpm = bpm

                    graph = draw_graph(filtered, self.graph_width, self.graph_height, self.buffer_max_size)
                    bpm_display = draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.graph_height)

                    if self.show_fps:
                        view = draw_fps(view, fps)

                else:
                    graph, bpm_display = self.detected_gui(curr_buffer_size)

                # Show GUI in window
                graph = np.hstack((graph, bpm_display))
                view = np.vstack((view, graph))

            else:
                # No faces detected, so we must clear the lists of values and timestamps. Otherwise there will be a gap
                # in timestamps when a face is detected again.

                # GUI
                view = self.no_face_gui(view)

                # Clear values
                del self.values[:]
                del self.R[:]
                del self.G[:]
                del self.B[:]
                del self.times[:]

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27 or r == 0:
                break

    def run_pulse_observer_in_pos_method(self):
        cv2.namedWindow(self.window_name)

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(self.window_name, 0) == 0:
            r, frame = self.webcam.read()

            # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            # The original frame will be used to compute heart rate.
            view = np.array(frame)

            # Detect face using dlib
            faces = self.detector(frame, 0)
            if len(faces) == 1:

                # Heart rate graph gets 75% of window width.  BPM gets 25%.
                self.graph_width = int(view.shape[1] * 0.75)
                self.bpm_display_width = view.shape[1] - self.graph_width
                face_points = self.predictor(frame, faces[0])

                # Get the regions of interest.
                fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
                nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                # Slice out the regions of interest (ROI)
                self.fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                self.nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]

                # Average values and add to list
                r, g, b = return_avg(self.fh_roi, self.nose_roi)
                self.R.append(r)
                self.G.append(g)
                self.B.append(b)

                # Add time to list
                self.times.append(time.time())

                # Buffer is full, so pop the value off the top
                if len(self.times) > self.buffer_max_size:
                    self.R.pop(0)
                    self.G.pop(0)
                    self.B.pop(0)
                    self.times.pop(0)

                curr_buffer_size = len(self.times)

                # Don't try to compute pulse until we have at least the min. number of frames
                if curr_buffer_size > self.min_frames:
                    # Compute relevant times
                    time_elapsed = self.times[-1] - self.times[0]
                    fps = curr_buffer_size / time_elapsed  # frames per second

                    # Concatenate three channels to matrix
                    rgb = np.array([self.R, self.G, self.B])

                    # POS and normalize the output
                    filtered = sb_pos(rgb, curr_buffer_size)
                    filtered = normalize(filtered)
                    filtered = check_nan(filtered)

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(filtered))

                    # Generate list of frequencies that correspond to the FFT values
                    freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                    # because they correspond to impossible BPM values.
                    while True:
                        max_idx = fft.argmax()
                        bps = freqs[max_idx]
                        if bps < self.min_hz or bps > self.max_hz:
                            print('BPM of {0} was discarded.'.format(bps * 60.0))
                            fft[max_idx] = 0
                        else:
                            bpm = bps * 60.0
                            break

                    # It's impossible for the heart rate to change more than 10% between samples,
                    # so use a weighted average to smooth the BPM with the last BPM.
                    if self.last_bpm > 0:
                        bpm = (self.last_bpm * 0.9) + (bpm * 0.1)
                        self.last_bpm = bpm

                    graph = draw_graph(filtered, self.graph_width, self.graph_height, self.buffer_max_size)
                    bpm_display = draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.graph_height)

                    if self.show_fps:
                        view = draw_fps(view, fps)

                else:
                    graph, bpm_display = self.detected_gui(curr_buffer_size)

                # Show GUI in window
                graph = np.hstack((graph, bpm_display))
                view = np.vstack((view, graph))

            else:
                # No faces detected, so we must clear the lists of values and timestamps.
                # Otherwise there will be a gap in timestamps when a face is detected again.

                # GUI
                view = self.no_face_gui(view)

                # Clear values
                del self.values[:]
                del self.R[:]
                del self.G[:]
                del self.B[:]
                del self.times[:]

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27 or r == 0:
                break


def main():
    # Choose methods to estimate HR
    method = ['G',  # Not good
              'GRD',  # Relatively good
              'Adaptive GRD',  # Relatively good
              'ICA',  # Unstable
              'POS']  # Slow
    selected_method = method[0]

    # Initialize camera and detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    webcam = cv2.VideoCapture(0)

    # Start
    if webcam.isOpened():
        recorder = Recorder(detector, predictor, webcam)
        if selected_method == 'G':
            recorder.run_pulse_observer_in_g_method()
        elif selected_method == 'GRD':
            recorder.run_pulse_observer_in_grd_method()
        elif selected_method == 'Adaptive GRD':
            recorder.run_pulse_observer_in_adaptive_grd_method()
        elif selected_method == 'ICA':
            recorder.run_pulse_observer_in_ica_method()
        elif selected_method == 'POS':
            recorder.run_pulse_observer_in_pos_method()
        webcam.release()
    else:
        print('Failed to open your camera.')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
