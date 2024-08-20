import torch
from torch.utils.data import DataLoader

from kon_sequencer.paths import *
from kon_sequencer.params import *
from kon_sequencer.data_modules import  GlobalTempoSampler, MultiTrackDataset

from kon_sequencer.sequencer import KonSequencer

tempo_sampler = GlobalTempoSampler(TEMPO_LOW, TEMPO_HIGH)
kick_samples, kick_sample_rates = MultiTrackDataset.load_wav_folder(KK_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH) # Replace with actual paths to your one-shot samples
snare_samples, snare_sample_rates = MultiTrackDataset.load_wav_folder(SN_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH) # Replace with actual paths to your one-shot samples

kick_and_snare = MultiTrackDataset(tempo_sampler = tempo_sampler, one_shot_samples_list = [kick_samples, snare_samples], sample_rates_list = [kick_sample_rates, snare_sample_rates], num_steps=NUM_STEPS, num_tracks = NUM_TRACKS) 

#Test data loader
data_loader = DataLoader(kick_and_snare, batch_size=BATCH_SIZE, shuffle=True)

#Get an iterator from the DataLoader
data_iter = iter(data_loader)

#Get the first batch
batch = next(data_iter)

print("Batch first dimension shape:", batch[0].shape)
print("Batch second dimension shape:", batch[1].shape)
print("Batch third dimension shape:", batch[2].shape)

samples, step_vectors, tempos = next(data_iter)
print("Batch first dimension shape:", samples.shape)
print("Batch second dimension shape:", step_vectors.shape)
print("Batch third dimension shape:", tempos.shape)

sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)

#one data point with multiple tracks
for samples, step_vectors, tempo in zip(samples, step_vectors, tempos):
    print("Samples shape of one data point:", samples.shape) #torch.Size([2, 1, 12800])
    print("Step vectors shape of one data point:", step_vectors.shape) #torch.Size([2, 8])
    print("Tempo:", tempo)

    tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)
    print("Tracks shape:", tracks.shape) #torch.Size([2, 1, 64000])
    sequencer.save_multi_tracks(tracks, samples, step_vectors, tempo, OUT_DIR)
    break

"""
    #synthesize one track, is working
    for sample, step_vector in zip(samples, step_vectors):
        one_track = sequencer.render_one_track(sample, step_vector, tempo)
        #save to out, with tempo and step vectors info, and listen
        sequencer.save_one_track(one_track, sample, step_vector, tempo, OUT_DIR)
        break
    #break
    
"""

# next: process the batch so that it synthesizes a batch of loops, with shape (batch_size, 1, loop_length)
# be careful about the last step in the loop, it should truncate the one-shot sample to be able to maintain 16000*4 total loop length
