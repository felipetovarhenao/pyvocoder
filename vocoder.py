import numpy as np
import librosa
import soundfile as sf

PI: float = np.pi
TWO_PI: float = PI * 2


class PV:

    def __init__(self, hop_length: int = 128, win_length: int = 1024) -> None:
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.nyquist_index: int = self.win_length // 2 + 1
        self.windowing_function: np.ndarray = np.hanning(self.win_length)
        self.bin_indices = np.arange(self.nyquist_index)
        self.bin_frequencies: np.ndarray = TWO_PI * self.bin_indices / self.win_length
        self.prev_input_phase: np.ndarray = np.zeros(self.nyquist_index)
        self.prev_output_phase: np.ndarray = np.zeros(self.nyquist_index)

    def retune(self, path: str, semitones: float = 220) -> tuple[np.ndarray, int]:
        input_signal, sr = librosa.load(path=path, sr=None, mono=True)

        num_frames: int = (len(input_signal) - self.win_length) // self.hop_length

        # output signal array
        output_signal: np.ndarray = np.zeros_like(input_signal)

        detected_frequencies = librosa.yin(y=input_signal,
                                           fmin=50,
                                           fmax=1200,
                                           frame_length=self.win_length,
                                           hop_length=self.hop_length)[:num_frames]

        factor = 2**(semitones/12)
        quantized_frequencies = self.quantize_frequencies(self.smooth_array(detected_frequencies) * factor)

        for i in range(num_frames):
            st: int = i * self.hop_length
            end: int = st + self.win_length
            input_frame: np.ndarray = input_signal[st:end] * self.windowing_function

            input_mag, input_frequencies = self.analyze_frame(frame=input_frame)

            base_frequency = quantized_frequencies[i]
            base_radian_frequency = (base_frequency/sr) * TWO_PI

            output_signal[st:end] += self.resynth_frame(input_mag=input_mag,
                                                        sr=sr,
                                                        input_frequencies=input_frequencies,
                                                        base_frequency=base_frequency,
                                                        base_radian_frequency=base_radian_frequency)

        return output_signal, sr

    def analyze_frame(self, frame):
        input_spectrum: np.ndarray = np.fft.fft(a=frame)
        input_mag: np.ndarray = self.split_frame(np.abs(input_spectrum))
        input_phase: np.ndarray = self.split_frame(np.angle(input_spectrum))

        phase_diff: np.ndarray = input_phase - self.prev_input_phase
        phase_diff = self.wrap_phase(phase_diff - self.bin_frequencies * self.hop_length)

        frequency_deviation: np.ndarray = phase_diff / self.hop_length

        input_frequencies: np.ndarray = self.bin_frequencies + frequency_deviation
        self.prev_input_phase = input_phase
        return input_mag, input_frequencies

    def resynth_frame(self, input_mag, sr,  input_frequencies, base_frequency, base_radian_frequency):
        output_mag: np.ndarray = np.zeros(self.nyquist_index)
        output_frequencies: np.ndarray = np.zeros(self.nyquist_index)

        harmonic_raw = (input_frequencies * sr) / (TWO_PI * base_frequency)
        harmonic_below = np.floor(harmonic_raw)
        harmonic_above = harmonic_below + 1
        harmonic_fraction = harmonic_raw-harmonic_below

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

        # —————————— RESYNTH STAGE ————————————
        frequency_deviation = output_frequencies - self.bin_frequencies
        phase_diff = frequency_deviation * self.hop_length

        phase_diff += self.bin_frequencies * self.hop_length

        output_phase = self.wrap_phase(self.prev_output_phase + phase_diff)

        self.prev_output_phase = output_phase

        # resynthesis
        output_mag = self.mirror_frame(output_mag)
        output_phase = self.mirror_frame(output_phase, is_phase=True)
        output_spectrum = output_mag * np.exp(1j * output_phase)
        output_frame = np.real(np.fft.ifft(a=output_spectrum))
        return output_frame * self.windowing_function

    def quantize_frequencies(self, frequencies, pitch_class_list=np.array([0, 3, 7])):
        pitches = np.zeros_like(frequencies)
        positive_indices = np.argwhere(frequencies > 0)[:, 0]
        pitches[positive_indices] = self.freq_to_pitch(frequencies[positive_indices])
        quantized_pitches = np.array([self.quantize_pitch(p, pitch_class_list) for p in pitches])
        quantized_frequencies = np.zeros_like(frequencies)
        quantized_frequencies[positive_indices] = self.pitch_to_freq(quantized_pitches[positive_indices])
        return quantized_frequencies

    def quantize_pitch(self, pitch: float, pitch_grid):
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


p = PV()
data, samplerate = p.retune("./input/i can hear your voice mid h1.wav", 0)
sf.write('./output/output.wav', data=data, samplerate=samplerate)
