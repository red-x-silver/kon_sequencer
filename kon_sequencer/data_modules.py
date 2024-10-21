import torch
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

    
#no use now
"""
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
"""   

    
class OneShotSamplesDataset():
    def __init__(self, one_shots_input_dir, make_mono=True, unifyLenth = True, targetLength = 16000, ext = ["wav"]):
        self.input_dir = one_shots_input_dir #input_dir is the directory containing all the one-shot samples of a INGLE TRACK
        self.one_shot_waveforms = []
        self.one_shot_sample_rates = []
        self.one_shot_sample_names = []
        self.make_mono = make_mono
        self.unifyLenth = unifyLenth
        self.targetLength = targetLength
        self.targetExt = ext
        self.load_one_shot_samples(self.input_dir)
        
    def load_one_shot_samples(self):
        #for possibly nested folder
        for root_dir, _, file_names in os.walk(self.input_dir):
            for file_name in file_names:
                if file_name.endswith(tuple(self.targetExt)) and not file_name.startswith("."):
                    file_path = os.path.join(root_dir, file_name)
                    # Load the .wav file into a PyTorch tensor
                    waveform, sample_rate = torchaudio.load(file_path)
                    self.one_shot_sample_rates.append(sample_rate)
                    self.one_shot_sample_names.append(file_name)
                    if self.make_mono:
                        if waveform.shape[0] == 2:
                            waveform = (waveform[0] + waveform[1]) / 2
                    #unify every one-shot sample's length to be the same, for easy batch processing
                    if self.unifyLenth == True:
                        if waveform.shape[1] < self.targetLength:
                            pad = (self.targetLength - waveform.shape[1], 0)
                            waveform = F.pad(waveform, pad, "constant", 0)
                        else:
                            waveform = waveform[:,:self.targetLength] #waveform is of shape (1,num_samples) first dimension is for mono/stero channel
                    self.one_shot_waveforms.append(waveform)       
        return
    
    def __len__(self):
        assert len(self.one_shot_waveforms) == len(self.one_shot_sample_rates) == len(self.one_shot_sample_names)

        return len(self.one_shot_sample_names)
    
class MultiTrackDataset(Dataset):
    def __init__(self, tempo_sampler, oneShotSamplesDataset_Lists, num_steps=8, num_tracks = 3, dataset_size=100000):
        super(MultiTrackDataset, self).__init__()
        #one_shot_datasets_list are a list of customised object OneShotSamplesDataset, in which the waveforms are in tensors already read by torchaudio.load()
        self.one_shot_datasets_list = oneShotSamplesDataset_Lists
        self.num_steps = num_steps
        self.num_tracks = num_tracks
        self.tempo_sampler = tempo_sampler
        self.dataset_size = dataset_size

        assert len(self.one_shot_datasets_list) == self.num_tracks

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        tempo = self.tempo_sampler.generate_tempo()
        samples = []
        step_vectors = []
        sample_names = []

        for singleTrack_one_shot_dataset in self.one_shot_datasets_list:
             # Here we manually iterate over the dataset with the tempo input
            # Randomly select a one-shot sample
            index = random.randint(0, len(singleTrack_one_shot_dataset) - 1)
            sample = singleTrack_one_shot_dataset.one_shot_waveforms[index]
            samples.append(sample)
            sample_names.append(singleTrack_one_shot_dataset.one_shot_sample_names[index])

            # Randomly generate a step vector
            step_vector = torch.randint(0, 2, (self.num_steps,))
            step_vectors.append(step_vector)
            
        # Return the sample, step_vector, and externally provided tempo
        return torch.stack(samples, dim=0), torch.stack(step_vectors, dim=0), tempo, sample_names
 
    
"""
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
"""






    

