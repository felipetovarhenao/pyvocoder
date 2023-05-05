import numpy as np
import librosa
import soundfile as sf

PI: float = np.pi
TWO_PI: float = PI * 2


class PV:

    def __init__(self, hop_length: int = 512, win_length: int = 2048) -> None:
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.nyquist_index: int = self.win_length // 2 + 1
        self.windowing_function: np.ndarray = np.hanning(self.win_length)
        self.bin_indices = np.arange(self.nyquist_index)
        self.bin_frequencies: np.ndarray = TWO_PI * self.bin_indices / self.win_length
        self.prev_input_phase: np.ndarray = np.zeros(self.nyquist_index)
        self.prev_output_phase: np.ndarray = np.zeros(self.nyquist_index)

    def retune(self, path: str, fq: float = 220) -> tuple[np.ndarray, int]:
        input_signal, sr = librosa.load(path=path, sr=None, mono=True)

        base_frequency = fq
        base_radian_frequency = (base_frequency/sr) * TWO_PI

        num_frames: int = (len(input_signal) - self.win_length) // self.hop_length

        # output signal array
        output_signal: np.ndarray = np.zeros_like(input_signal)

        for i in range(num_frames):
            st: int = i * self.hop_length
            end: int = st + self.win_length
            input_frame: np.ndarray = input_signal[st:end] * self.windowing_function

            input_mag, input_frequencies = self.analyze_frame(frame=input_frame)

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

        harmonic_raw = input_frequencies * sr / TWO_PI / base_frequency
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
        new_bin = np.round(
            new_frequency * (self.win_length / TWO_PI)).astype('int32')

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

    @staticmethod
    def split_frame(frame: np.ndarray) -> np.ndarray:
        return frame[:len(frame) // 2 + 1]

    @staticmethod
    def mirror_frame(frame: np.ndarray, is_phase: bool = False) -> np.ndarray:
        return np.concatenate([frame[:-1], (-1 if is_phase else 1) * np.flip(frame[1:])])

    @staticmethod
    def wrap_phase(phase: np.ndarray) -> np.ndarray:
        return np.mod(phase+PI, TWO_PI) - PI


p = PV()
data, samplerate = p.retune('./input/i-do-not-know-who-i-am.wav', 660)
sf.write('./output/output.wav', data=data, samplerate=samplerate)
