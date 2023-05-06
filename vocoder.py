import numpy as np
import librosa

PI: float = np.pi
TWO_PI: float = PI * 2


class PV:

    def __init__(self, hop_length: int = 128, win_length: int = 2048) -> None:
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.nyquist_index: int = self.win_length // 2 + 1
        self.windowing_function: np.ndarray = np.hanning(self.win_length)
        self.bin_indices = np.arange(self.nyquist_index)
        self.bin_frequencies: np.ndarray = TWO_PI * self.bin_indices / self.win_length
        self.prev_input_phase: np.ndarray = np.zeros(self.nyquist_index)
        self.prev_output_phases: np.ndarray | None = None

    def retune(self, path: str, pitch_list: list = [60, 64, 67]) -> tuple[np.ndarray, int]:
        """ Main function — applies robotization effect to input file and retunes it to `pitch_list` argument """
        # load input file as an array
        input_signal, sr = librosa.load(path=path, sr=None, mono=True)

        # number of overlapping FFT frames
        num_frames: int = (len(input_signal) - self.win_length) // self.hop_length

        # initialize empty output audio signal array
        output_signal: np.ndarray = np.zeros(shape=(len(input_signal), 2))

        # convert pitch collection to frequency
        frequency_list = self.pitch_to_freq(np.array(pitch_list))

        # shufle frequency array (for random stereo panning inside loop)
        np.random.shuffle(frequency_list)

        # initialize output phase buffer (one array per pitch)
        self.prev_output_phases: np.ndarray = np.zeros(shape=(len(frequency_list), self.nyquist_index))

        # analysis-resynthesis per FFT frame
        for i in range(num_frames):
            # get index range and extract windowed frame
            st: int = i * self.hop_length
            end: int = st + self.win_length
            input_frame: np.ndarray = input_signal[st:end] * self.windowing_function

            # get input magnitudes and frequencies from frame
            input_mag, input_frequencies = self.__analyze_frame(frame=input_frame)

            # apply resynthesis for each target pitch
            for n, base_frequency in enumerate(frequency_list):
                # frequency in radians
                base_radian_frequency = (base_frequency/sr) * TWO_PI

                # panning value based on voice index
                pan = n/(len(frequency_list)-1)
                pan = np.array([1-pan, pan])

                # get synthesis magnitudes and frequencies based on target pitch
                output_mag, output_frequencies = self.__process_frame(input_mag=input_mag,
                                                                      sr=sr,
                                                                      input_frequencies=input_frequencies,
                                                                      base_frequency=base_frequency,
                                                                      base_radian_frequency=base_radian_frequency)

                # resynthesize frame, apply panning, and add to output signal
                output_frame = self.__resynth_frame(output_mag, output_frequencies, voice_index=n)
                output_frame = np.repeat(output_frame[:, None], repeats=2, axis=1) * pan
                output_signal[st:end] += output_frame

        output_signal /= output_signal.max()
        return output_signal, sr

    def __analyze_frame(self, frame):
        # perform FFT analysis and get left half of magnitudes and phases
        input_spectrum: np.ndarray = np.fft.fft(a=frame)
        input_mag: np.ndarray = self.split_frame(np.abs(input_spectrum))
        input_phase: np.ndarray = self.split_frame(np.angle(input_spectrum))

        # get current phase difference between adjacent FFT frames and wrap to [-pi, pi) range
        phase_diff: np.ndarray = input_phase - self.prev_input_phase
        phase_diff = self.wrap_phase(phase_diff - self.bin_frequencies * self.hop_length)

        # get deviation from bins' center frequency
        frequency_deviation: np.ndarray = phase_diff / self.hop_length

        # get true frequencies
        input_frequencies: np.ndarray = self.bin_frequencies + frequency_deviation

        # keep track of phase for next FFT frame
        self.prev_input_phase = input_phase

        return input_mag, input_frequencies

    def __process_frame(self, input_mag, sr,  input_frequencies, base_frequency, base_radian_frequency):
        # initialize arrays for output magnitudes and frequencies
        output_mag: np.ndarray = np.zeros(self.nyquist_index)
        output_frequencies: np.ndarray = np.zeros(self.nyquist_index)

        # get float partial frequency by normalizing radians and mutiplying by fractional bin number of target frequency
        harmonic_raw = (input_frequencies / TWO_PI) * (sr / base_frequency)

        # get lower and upper bins, along with the fractional difference
        harmonic_below = np.floor(harmonic_raw)
        harmonic_above = harmonic_below + 1
        harmonic_fraction = harmonic_raw - harmonic_below

        # handle lower harmonic
        harmonic_below_indices = np.argwhere(harmonic_below > 0)[:, 0]
        filtered_indices = self.bin_indices[harmonic_below_indices]

        new_frequency = base_radian_frequency * harmonic_below[harmonic_below_indices]
        new_bin = np.round(new_frequency * (self.win_length / TWO_PI)).astype('int32')

        index_mask = np.argwhere(new_bin < self.nyquist_index)[:, 0]
        indices = filtered_indices[index_mask]

        new_bin = new_bin[index_mask]
        output_mag[new_bin] += input_mag[indices] * (1-harmonic_fraction[indices])
        output_frequencies[new_bin] = new_frequency[index_mask]

        # handle upper harmonic
        new_frequency = base_radian_frequency * harmonic_above
        new_bin = np.round(new_frequency * (self.win_length / TWO_PI)).astype('int32')

        index_mask = np.argwhere(new_bin < self.nyquist_index)[:, 0]
        indices = self.bin_indices[index_mask]

        new_bin = new_bin[index_mask]
        output_mag[new_bin] += input_mag[indices] * harmonic_fraction[indices]
        output_frequencies[new_bin] = new_frequency[index_mask]
        return output_mag, output_frequencies

    def __resynth_frame(self, output_mag, output_frequencies, voice_index):
        # —————————— RESYNTH STAGE ————————————
        frequency_deviation = output_frequencies - self.bin_frequencies
        phase_diff = frequency_deviation * self.hop_length

        phase_diff += self.bin_frequencies * self.hop_length

        output_phase = self.wrap_phase(self.prev_output_phases[voice_index] + phase_diff)

        self.prev_output_phases[voice_index] = output_phase

        output_mag = self.mirror_frame(output_mag)
        output_phase = self.mirror_frame(output_phase, is_phase=True)
        output_spectrum = output_mag * np.exp(1j * output_phase)
        output_frame = np.real(np.fft.ifft(a=output_spectrum))
        return output_frame * self.windowing_function

    def __quantize_frequencies(self, frequencies, pitch_class_list=np.array([0, 3, 7])):
        """ ignore this private method """
        pitches = np.zeros_like(frequencies)
        positive_indices = np.argwhere(frequencies > 0)
        pitches[positive_indices] = self.freq_to_pitch(frequencies[positive_indices])
        quantized_pitches = np.array([self.__quantize_pitch(p, pitch_class_list) for p in pitches])
        quantized_frequencies = np.zeros_like(frequencies)
        quantized_frequencies[positive_indices] = self.pitch_to_freq(quantized_pitches[positive_indices])
        return quantized_frequencies

    def __quantize_pitch(self, pitch: float, pitch_grid):
        if pitch == 0:
            return pitch
        pitch_class = pitch % 12
        pitch_dist = np.minimum(np.abs(pitch_grid - pitch_class), 12 - np.abs(pitch_grid - pitch_class))
        nearest_pitch_class = pitch_grid[np.argmin(pitch_dist)]
        return np.round(pitch - pitch_class + nearest_pitch_class)

    @ staticmethod
    def split_frame(frame: np.ndarray) -> np.ndarray:
        return frame[:len(frame) // 2 + 1]

    @ staticmethod
    def mirror_frame(frame: np.ndarray, is_phase: bool = False) -> np.ndarray:
        return np.concatenate([frame[:-1], (-1 if is_phase else 1) * np.flip(frame[1:])])

    @ staticmethod
    def wrap_phase(phase: np.ndarray) -> np.ndarray:
        return np.mod(phase+PI, TWO_PI) - PI

    @ staticmethod
    def freq_to_pitch(freq_array):
        return 12 * np.log2(freq_array / 440) + 69

    @ staticmethod
    def pitch_to_freq(pitch_array):
        return 440 * 2 ** ((pitch_array - 69) / 12)

    @ staticmethod
    def smooth_array(arr, window_size: int = 11):
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")

        # Create a new array to hold the smoothed values
        smoothed_arr = np.zeros_like(arr)

        # Apply the smoothing filter to each element in the array
        for i in range(len(arr)):
            left_idx = max(0, i - (window_size-1)//2)
            right_idx = min(len(arr)-1, i + (window_size-1)//2)
            smoothed_arr[i] = np.mean(arr[left_idx:right_idx+1])

        return smoothed_arr
