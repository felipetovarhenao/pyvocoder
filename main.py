from vocoder import PV
import soundfile as sf

p = PV()
data, samplerate = p.retune("./input/i-do-not-know-who-i-am.wav", pitch_list=[60, 63.5, 67, 69.4])
sf.write('./output/output.wav', data=data, samplerate=samplerate)
