import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

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

"""
dev code for checking the shape of one batch
batch = next(data_iter)
print("Batch first dimension shape:", batch[0].shape)
print("Batch second dimension shape:", batch[1].shape)
print("Batch third dimension shape:", batch[2].shape)
"""

"""
dev code for generating each data points within one batch 

samples, step_vectors, tempos = next(data_iter)
print("Batch first dimension shape:", samples.shape)
print("Batch second dimension shape:", step_vectors.shape)# torch.Size([16, 2, 8])
print("Batch third dimension shape:", tempos.shape) #shape: torch.Size([16, 1])

sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)

#one data point with multiple tracks
for samples, step_vectors, tempo in zip(samples, step_vectors, tempos):
    print("Samples shape of one data point:", samples.shape) #torch.Size([2, 1, 12800])
    print("Step vectors shape of one data point:", step_vectors.shape) #torch.Size([2, 8])
    print("Tempo:", tempo) 

    tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)
    print("Tracks shape:", tracks.shape)
    sequencer.save_multi_tracks(tracks, samples, step_vectors, tempo, OUT_DIR)
    break
"""
sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)

"""
dev code for generating target vectors and neural network inputs aka synthesized loops for each batch

samples, step_vectors, tempos = next(data_iter)
# Combine tempos and step_vectors to form the target
#print("step_vectors shape:", step_vectors.shape) #shape: torch.Size([16, 2, 8])
#step vectors: batch, num_tracks, num_steps -> batch, num_tracks*num_steps
print("step vectors:", step_vectors)

target_step_vectors = step_vectors.view(step_vectors.shape[0], -1)
#print("Target step vectors shape after torch.cat along dim 1:", target_step_vectors.shape) #torch.Size([16, 16])
print("step vectors after re-shaping:", target_step_vectors)

#tempo: batch, 1
#target: batch, 1 + num_tracks*num_steps
targets = torch.cat([tempos, target_step_vectors], dim=1) 
#print("Targets shape after torch.cat tempo and step vectors along dim 1:", targets.shape) # torch.Size([16, 17])
print("tempos::", tempos)
print("final target vectors", targets)

synthesized_loops = []

for samples, step_vectors, tempo in zip(samples, step_vectors, tempos):
    #print("Samples shape of one data point:", samples.shape) #torch.Size([2, 1, 12800])
    #print("Step vectors shape of one data point:", step_vectors.shape) #torch.Size([2, 8])
    #print("Tempo:", tempo)

    multi_tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)  #Tracks shape: torch.Size([2,1,64000]), no batch dim here
    sum_track = torch.sum(multi_tracks, dim=0)/2 #shape: torch.Size([1, 64000])
    synthesized_loops.append(sum_track) #shape: torch.Size([1, 64000])

synthesized_loops_batch = torch.stack(synthesized_loops, dim=0)#batch, 1 for mono, sample_length
print("Synthesized loops batch shape:", synthesized_loops_batch.shape)
"""


"""
    #synthesize one track, is working
    for sample, step_vector in zip(samples, step_vectors):
        one_track = sequencer.render_one_track(sample, step_vector, tempo)
        #save to out, with tempo and step vectors info, and listen
        sequencer.save_one_track(one_track, sample, step_vector, tempo, OUT_DIR)
        break
    #break
    
"""

# next: process the batch so that it synthesizes a batch of loops, with shape (batch_size, 1, loop_length) done!
# be careful about the last step in the loop (and beginning), it should truncate the one-shot sample to be able to maintain 16000*4 total loop length done!

#to check: sounds like some one-shot start quite late in its attack phase



class KonSequencerModel(pl.LightningModule):
    def __init__(self, sample_rate, num_steps=8):
        super(KonSequencerModel, self).__init__()
        self.sample_rate = sample_rate
        self.num_steps = num_steps
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * num_steps, 64)
        self.fc2 = nn.Linear(64, 9)  # 1 for tempo + 8 for step vector

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        samples, step_vectors, tempos = batch
        # Combine tempos and step_vectors to form the target
        target_step_vectors = step_vectors.view(step_vectors.shape[0], -1)
        #print("Target step vectors shape after torch.cat along dim 1:", target_step_vectors.shape) #torch.Size([16, 16])
        #tempo: batch, 1
        #target: batch, 1 + num_tracks*num_steps
        targets = torch.cat([tempos, target_step_vectors], dim=1) 

        #synthesize batch_size loops
        synthesized_loops = []
        #process one data point by one data point where one data point comprises multi tracks
        for samples, step_vectors, tempo in zip(samples, step_vectors, tempos):
            #print("Samples shape of one data point:", samples.shape) #torch.Size([2, 1, 12800])
            #print("Step vectors shape of one data point:", step_vectors.shape) #torch.Size([2, 8])
            #print("Tempo:", tempo) 

            multi_tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)  #Tracks shape: torch.Size([2,1,64000]), no batch dim here
            sum_track = torch.sum(multi_tracks, dim=0)/2 #shape: torch.Size([1, 64000])
            synthesized_loops.append(sum_track) #shape: torch.Size([1, 64000])

        #input to the neural network
        synthesized_loops_batch = torch.stack(synthesized_loops, dim=0)#batch, 1 for mono, sample_length
        print("Synthesized loops batch shape:", synthesized_loops_batch.shape) #shape: torch.Size([16, 1, 64000])


        # Predict tempo and step vector
        y_pred = self(synthesized_loops_batch)

        # Loss is computed separately for tempo and step vector
        tempo_loss = nn.functional.mse_loss(y_pred[:, 0], targets[:, 0])
        step_vector_loss = nn.functional.binary_cross_entropy_with_logits(y_pred[:, 1:], targets[:, 1:])

        loss = tempo_loss + step_vector_loss

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
