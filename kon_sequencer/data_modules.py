import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
import random
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F


class GlobalTempoSampler:
    def __init__(self, tempo_low, tempo_high):
        self.tempo_low = tempo_low
        self.tempo_high = tempo_high


    def generate_tempo(self):
        # Generate a global tempo to be shared across datasets
        return torch.randint(self.tempo_low, self.tempo_high, (1,))

    

class SingleTrackDataset(Dataset):
    def __init__(self, tempo_sampler, one_shot_samples, sample_rates, num_steps=8):
        super(SingleTrackDataset, self).__init__()
        #one_shot_samples are list of tensors already read by torchaudio.load()
        self.one_shot_samples = one_shot_samples
        self.sample_rates = sample_rates
        self.num_steps = num_steps
        self.tempo_sampler = tempo_sampler

    def __len__(self):
        return 10000  # Simulate a large dataset

    def __getitem__(self, idx):
        tempo = self.tempo_sampler.generate_tempo()

        # Randomly select a one-shot sample
        sample = random.choice(self.one_shot_samples)

        # Randomly generate a step vector
        step_vector = torch.randint(0, 2, (self.num_steps,))

        # Return the sample, step_vector, and externally provided tempo
        return sample, step_vector, tempo
    
    @staticmethod
    def load_wav_folder(input_dir,make_mono=True, unifyLenth = True, targetLength = 16000, ext = "wav"):
        waveforms = []
        sample_rates = []
        #for possibly nested folder
        for root_dir, _, file_names in os.walk(input_dir):
            for file_name in file_names:
                if file_name.endswith(ext) and not file_name.startswith("."):
                    file_path = os.path.join(root_dir, file_name)
                    # Load the .wav file into a PyTorch tensor
                    waveform, sample_rate = torchaudio.load(file_path)
                    if make_mono:
                        if waveform.shape[0] == 2:
                            waveform = (waveform[0] + waveform[1]) / 2
                 #unify every one-shot sample's length to be the same
                    if unifyLenth == True:
                        if waveform.shape[1] < targetLength:
                            pad = (targetLength - waveform.shape[1], 0)
                            waveform = F.pad(waveform, pad, "constant", 0)
                        else:
                            waveform = waveform[:,:targetLength]
                    waveforms.append(waveform)
                    sample_rates.append(sample_rate)

        return waveforms, sample_rates
    
class MultiTrackDataset(Dataset):
    def __init__(self, tempo_sampler, one_shot_samples_list, sample_rates_list, num_steps=8, num_tracks = 3):
        super(MultiTrackDataset, self).__init__()
        #one_shot_samples are list of tensors already read by torchaudio.load()
        self.one_shot_samples_list = one_shot_samples_list
        self.sample_rates_list = sample_rates_list
        self.num_steps = num_steps
        self.num_tracks = num_tracks
        self.tempo_sampler = tempo_sampler

        assert len(self.one_shot_samples_list) == self.num_tracks

    def __len__(self):
        return 100000  # Simulate a large dataset
    
    def __getitem__(self, idx):
        tempo = self.tempo_sampler.generate_tempo()
        samples = []
        step_vectors = []

        for singleTrack_one_shot_samples in self.one_shot_samples_list:
             # Here we manually iterate over the dataset with the tempo input
            # Randomly select a one-shot sample
            sample = random.choice(singleTrack_one_shot_samples)
            samples.append(sample)

            # Randomly generate a step vector
            step_vector = torch.randint(0, 2, (self.num_steps,))
            step_vectors.append(step_vector)
            
        # Return the sample, step_vector, and externally provided tempo
        return torch.stack(samples, dim=0), torch.stack(step_vectors, dim=0), tempo
    
    @staticmethod
    def load_wav_folder(input_dir,make_mono=True, unifyLenth = True, targetLength = 16000, ext = "wav"):
        waveforms = []
        sample_rates = []
        #for possibly nested folder
        for root_dir, _, file_names in os.walk(input_dir):
            for file_name in file_names:
                if file_name.endswith(ext) and not file_name.startswith("."):
                    file_path = os.path.join(root_dir, file_name)
                    # Load the .wav file into a PyTorch tensor
                    waveform, sample_rate = torchaudio.load(file_path)
                    if make_mono:
                        if waveform.shape[0] == 2:
                            waveform = (waveform[0] + waveform[1]) / 2
                 #unify every one-shot sample's length to be the same
                    if unifyLenth == True:
                        if waveform.shape[1] < targetLength:
                            pad = (targetLength - waveform.shape[1], 0)
                            waveform = F.pad(waveform, pad, "constant", 0)
                        else:
                            waveform = waveform[:,:targetLength] #waveform is of shape (1,num_samples) first dimension is for mono/stero channel
                    waveforms.append(waveform)
                    sample_rates.append(sample_rate)

        return waveforms, sample_rates
    

class MultiTrackDataLoader:
    def __init__(self, datasets, batch_size, tempo_sampler, number_of_tracks = 3):
        self.datasets = datasets
        self.batch_size = batch_size
        self.tempo_sampler = tempo_sampler

        assert len(self.datasets) == number_of_tracks

    def __iter__(self):
        combined_batch = []
        #one data point[[3xsamples],[3xstep vectors],tempo]
        for i in range(self.batch_size):
           #sample to be one second long uniformly
            #step vector to be 8 steps long uniformly
            tempo = self.tempo_sampler.generate_tempo()
            one_data_point = [[],[],tempo]

            for singleTrackDataset in self.datasets:
                 # Here we manually iterate over the dataset with the tempo input
                sample, step_vector = singleTrackDataset[i]
                one_data_point[0].append(sample)
                one_data_point[1].append(step_vector)
            combined_batch.append(one_data_point)
        yield combined_batch







    

