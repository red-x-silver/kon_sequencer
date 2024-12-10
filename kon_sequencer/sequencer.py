#%%
import torch
import torchaudio
import torch.nn.functional as F
import os

#difference between data_modules and sequencer is that data_modules is for data processing done on CPU, sequencer is for audio processing better to be done on GPU

#%%
class KonSequencer:
    def __init__(self, num_tracks, num_steps, steps_per_beat, sample_rate, loop_length, left_safety_padding, right_safety_padding):
        self.num_tracks = num_tracks
        self.num_steps = num_steps
        self.steps_per_beat = steps_per_beat
        self.sample_rate = sample_rate
        self.loop_length = loop_length
        self.left_safety_padding = left_safety_padding #for the activation vector
        self.right_safety_padding = right_safety_padding #for the activation vector



    def calculate_samples_per_step(self, tempo):
        # Calculate the number of samples per beat
        samples_per_beat = torch.floor(self.sample_rate * 60 / tempo)
        samples_per_step = torch.floor(samples_per_beat / self.steps_per_beat)
        return int(samples_per_step)
    

    def generate_onetrack_activation_vector(self, step_vector, tempo):
        #step_vector shape: torch.Size([8])
        samples_per_step = self.calculate_samples_per_step(tempo)
        activation_vector = torch.zeros(self.loop_length)
        for idx, active in enumerate(step_vector):
            if active:
                activation_vector[idx*samples_per_step] = 1
        return activation_vector
    
    def generate_multitrack_activation_vectors(self, step_vectors, tempo):
         #step_vector shape: torch.Size([2, 8])
        samples_per_step = self.calculate_samples_per_step(tempo)
        num_tracks = step_vectors.shape[0]
        activation_vectors = torch.zeros(num_tracks, self.loop_length)
        for track_idx, step_vector in enumerate(step_vectors):
            #step_vector shape: torch.Size([8])
            for idx, active in enumerate(step_vector):
                if active:
                    activation_vectors[track_idx,idx*samples_per_step] = 1
        return activation_vectors
    
    def render_multi_tracks(self, one_shot_samples, step_vectors, tempo):
        activation_vectors = self.generate_multitrack_activation_vectors(step_vectors, tempo)
        #print(f"activation_vectors shape: {activation_vectors.shape}") #shape: torch.Size([2, 64000])
        pad = (self.left_safety_padding, self.right_safety_padding)
        activation_vectors_padded = F.pad(activation_vectors, pad, "constant", 0)
        #print(f"activation_vectors_padded shape: {activation_vectors_padded.shape}") #shape: torch.Size([2,81599])
        #print(f"one_shot_samples shape: {one_shot_samples.shape}") #shape: torch.Size([2, 1, 12800])

        #activation_vectors_padded.unsqueeze(0) is to create the batch channel
        #
        tracks = F.conv1d(activation_vectors_padded.unsqueeze(0), one_shot_samples.flip(-1), padding = 0, groups = self.num_tracks)
        #print(f"right after conv1d tracks shape: {tracks.shape}") 
        #shape: torch.Size([1, 2, 68800]) first is the batch dimension, second is the track dimension, third is the time dimension
        tracks = tracks.squeeze(0).unsqueeze(1)
        #print(f"after reorganising dim to be track_dim(2 for 2 instruments),channel_dim(1 for mono),time_dim(num_samples): {tracks.shape}") 
        return tracks[:,:,:self.loop_length]
    """
    def save_multi_tracks(self, multi_tracks, one_shot_samples, tempo, output_dir, instru_names = ["kick", "snare", "hihats"], save_multitracks = False, stereo = False):
        #Tracks shape: torch.Size([2,1,64000])  #no batch dim here

        if save_multitracks:
            for idx, one_track in enumerate(multi_tracks):
                track_save_path = f"{output_dir}/{instru_names[idx]}.wav"
                torchaudio.save(track_save_path, one_track, sample_rate=self.sample_rate)

                oneshot_save_path = f"{output_dir}/one_shot_{instru_names[idx]}.wav"
                torchaudio.save(oneshot_save_path, one_shot_samples[idx], sample_rate=self.sample_rate)
        
        sum_track = torch.sum(multi_tracks, dim=0)/len(multi_tracks)
        print(f"sum_track shape: {sum_track.shape}") #shape: torch.Size([1, 64000])
        sum_track_save_path = f"{output_dir}/sum_track_bpm{tempo}.wav"
        if stereo:
            sum_track = sum_track.repeat(2, 1) 
        torchaudio.save(sum_track_save_path, sum_track, sample_rate=self.sample_rate)
    """


    def render_one_track(self, one_shot_sample, step_vector, tempo):
        #NOT batch operation, one sample, one step vector at a time.

        #step_vector shape: torch.Size([8])
        #one_shot_sample shape: torch.Size([1, 12800])
        #tempo: tensor(an integer)
        activation_vector = self.generate_onetrack_activation_vector(step_vector, tempo)
        print(f"activation_vector shape: {activation_vector.shape}") #shape: torch.Size([64000])
        pad = (self.left_safety_padding, self.right_safety_padding)
        activation_vector_padded = F.pad(activation_vector, pad, "constant", 0)
        print(f"activation_vector_padded shape: {activation_vector_padded.shape}")  #shape: torch.Size([81599])
        print(f"one_shot_sample shape: {one_shot_sample.shape}") #shape: torch.Size([1, 12800])
        #be careful of how many dimensions to unsqueeze, re the shape of activation vector and one shot samples, conv1d need 3 dimensions
        #also be careful of which dimension of one_shot_sample to flip, it should be the last (time) dimension
        track = F.conv1d(activation_vector_padded.unsqueeze(0).unsqueeze(0), one_shot_sample.flip(-1).unsqueeze(0), padding = 0)
        print(f"track shape: {track.shape}") #shape: torch.Size([1, 1, 68800])

        #todo: truncate track to remove the extra safety padding bits
        return track[:,:,:self.loop_length]
    
    def save_one_track(self, one_track, one_shot, step_vector, tempo, output_dir):
        track_save_path = f"{output_dir}/track_{tempo}_{step_vector}.wav"
        torchaudio.save(track_save_path, one_track.squeeze(0), sample_rate=self.sample_rate)

        oneshot_save_path = f"{output_dir}/one_shot_{tempo}_{step_vector}.wav"
        torchaudio.save(oneshot_save_path, one_shot, sample_rate=self.sample_rate)




"""
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
"""
