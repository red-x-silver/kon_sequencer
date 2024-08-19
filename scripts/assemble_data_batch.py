import torch
from torch.utils.data import DataLoader

from paths import *
from params import *
from kon_sequencer.data_modules import  GlobalTempoSampler, SingleTrackDataset, MultiTrackDataset

tempo_sampler = GlobalTempoSampler(TEMPO_LOW, TEMPO_HIGH)
kick_samples, kick_sample_rates = MultiTrackDataset.load_wav_folder(KK_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH) # Replace with actual paths to your one-shot samples
snare_samples, snare_sample_rates = MultiTrackDataset.load_wav_folder(SN_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH) # Replace with actual paths to your one-shot samples

kick_and_snare = MultiTrackDataset(tempo_sampler = tempo_sampler, one_shot_samples_list = [kick_samples, snare_samples], sample_rates_list = [kick_sample_rates, snare_sample_rates], num_steps=NUM_STEPS, num_tracks = NUM_TRACKS) 

#test data loader
data_loader = DataLoader(kick_and_snare, batch_size=BATCH_SIZE, shuffle=True)

# Get an iterator from the DataLoader
data_iter = iter(data_loader)

# Get the first batch
batch = next(data_iter)

print("Batch first dimension shape:", batch[0].shape)
print("Batch second dimension shape:", batch[1].shape)
print("Batch third dimension shape:", batch[2].shape)

