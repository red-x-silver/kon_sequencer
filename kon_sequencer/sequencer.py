#%%
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt

#%%

#input activation step and tempo, output activation vector
def generate_activation_vector(step_vector, tempo = 100, step_per_beat = 2, sample_rate=44100):
    # Calculate the number of samples per beat
    samples_per_beat = int(sample_rate * 60 / tempo) #lost some samples in the end if it is not an integer
    samples_per_step = samples_per_beat // step_per_beat

    # Generate the activation vector
    number_of_steps = len(step_vector)
    activation_vector = torch.zeros(number_of_steps * samples_per_step)
    for idx, active in enumerate(step_vector):
        if active:
            activation_vector[idx*samples_per_step] = 1
    return activation_vector

# input one track's activation vector and one-shot audio sample, render one audio of one track
def render_track(activation_vector, audio_sample):
    # Generate the drum loop
    total_number_of_samples = activation_vector.shape[0]
    pad = (audio_sample.shape[0]-1, 0)
    activation_vector_padded = F.pad(activation_vector, pad, "constant", 0)
    track = F.conv1d(activation_vector_padded.unsqueeze(0).unsqueeze(0), audio_sample.flip(0).unsqueeze(0).unsqueeze(0), padding = 0)

    #track = F.conv1d(activation_vector.unsqueeze(0).unsqueeze(0), audio_sample.flip(0).unsqueeze(0).unsqueeze(0), padding = (audio_sample.shape[0]-1, ) )
    #track[:, :, :total_number_of_samples]
    return track

# input multiple tracks' audio samples and activation vectors, render the final audio
def sum_audio(tracks):
    # Initialize the final audio
    audio = torch.zeros_like(tracks[0])

    # Add the audio from each track
    for track in tracks:
        audio = audio + track

    return audio

def render_loop(step_vectors, audio_samples, tempo = 100, step_per_beat = 2, sample_rate=44100):
    # Generate the activation vectors
    activation_vectors = [generate_activation_vector(step_vector, tempo, step_per_beat, sample_rate) for step_vector in step_vectors]

    # Render the tracks
    tracks = [render_track(activation_vector, audio_sample) for activation_vector, audio_sample in zip(activation_vectors, audio_samples)]

    # Render the final audio
    audio = sum_audio(tracks)
    return audio

#%%
kick_path = "/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-kick.wav" # 909-kick.wav
snare_path = "/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-snare.wav" # 909-snare.wav
hh_path = "/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-hh.wav" # 909-hh.wav

kick = (torchaudio.load(kick_path)[0][0] + torchaudio.load(kick_path)[0][1])/2
snare= (torchaudio.load(snare_path)[0][0] + torchaudio.load(snare_path)[0][1])/2 #mono
hh = (torchaudio.load(hh_path)[0][0] + torchaudio.load(hh_path)[0][1])/2
#%%
print(kick.shape)
#%%
#kk_path = "/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-kick-mono.wav"
#torchaudio.save(kk_path, kick.unsqueeze(0), sample_rate=44100)
#%%
# Example activation vector
step_vector_kk = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
step_vector_sn = torch.tensor([0, 0, 1, 0, 0, 0, 0, 1])
step_vector_hh = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
#%%
act_vec_kk = generate_activation_vector(step_vector_kk)
act_vec_sn = generate_activation_vector(step_vector_sn)
act_vec_hh = generate_activation_vector(step_vector_hh)
activation_vectors = [act_vec_kk, act_vec_sn, act_vec_hh]
#%%
kk_seq = render_track(act_vec_kk, kick)
sn_seq = render_track(act_vec_sn, snare)
hh_seq = render_track(act_vec_hh, hh)
#%%
kk_seq_path = "/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-kick-seq.wav"
torchaudio.save(kk_seq_path, kk_seq.squeeze(0), sample_rate=44100)
#%%
_tempo = 145
loop = render_loop([step_vector_kk, step_vector_sn, step_vector_hh], [kick, snare, hh], tempo = _tempo)

save_path = f"/Users/xyi/Desktop/study2/kon-sequencer/one-shot-samples/909-loop-bpm{_tempo}.wav"
torchaudio.save(save_path, loop.squeeze(0), sample_rate=44100)

