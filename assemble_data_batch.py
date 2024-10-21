import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

from kon_sequencer.paths import *
from kon_sequencer.params import *
from kon_sequencer.data_modules import  GlobalTempoSampler, MultiTrackDataset, OneShotSamplesDataset

from kon_sequencer.sequencer import KonSequencer

from pytorch_lightning.loggers import WandbLogger


tempo_sampler = GlobalTempoSampler(TEMPO_LOW, TEMPO_HIGH)
#for training data 
kk_one_shots_dataset = OneShotSamplesDataset(KK_TRAIN_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
sn_one_shots_dataset = OneShotSamplesDataset(SN_TRAIN_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
hh_one_shots_dataset = OneShotSamplesDataset(HH_TRAIN_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
kk_snare_hh = MultiTrackDataset(tempo_sampler = tempo_sampler, oneShotSamplesDataset_Lists = [kk_one_shots_dataset, sn_one_shots_dataset, hh_one_shots_dataset], num_steps=NUM_STEPS, num_tracks = NUM_TRACKS,dataset_size=TRAIN_DATA_SIZE) 

#train data loader
train_data_loader = DataLoader(kk_snare_hh, batch_size=BATCH_SIZE, shuffle=True)

#for validation data
kk_one_shots_dataset_val = OneShotSamplesDataset(KK_VAL_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
sn_one_shots_dataset_val = OneShotSamplesDataset(SN_VAL_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
hh_one_shots_dataset_val = OneShotSamplesDataset(HH_VAL_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
kk_snare_hh_val = MultiTrackDataset(tempo_sampler = tempo_sampler, oneShotSamplesDataset_Lists = [kk_one_shots_dataset_val, sn_one_shots_dataset_val, hh_one_shots_dataset_val], num_steps=NUM_STEPS, num_tracks = NUM_TRACKS,dataset_size=TRAIN_DATA_SIZE) 
#val data loader
val_data_loader = DataLoader(kk_snare_hh_val, batch_size=BATCH_SIZE, shuffle=True)

#Get an iterator from the DataLoader
#data_iter = iter(train_data_loader)




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
#sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)

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
sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)
    #synthesize one track, is working
data_iter = iter(train_data_loader)
for i in range(1):
    samples, step_vectors, tempos = next(data_iter)
    print("Samples shape of one batch:", samples.shape) #torch.Size([2, 1, 12800])
    print("Step vectors shape of one batch:", step_vectors.shape) #torch.Size([2, 8])
    print("Tempo of one batch:", tempos)
    for samples, step_vectors, tempo in zip(samples, step_vectors, tempos):

        print("Samples shape of one data point:", samples.shape) #torch.Size([2, 1, 12800])
        print("Step vectors shape of one data point:", step_vectors.shape) #torch.Size([2, 8])
        print("Tempo of one data point:", tempo)
        #multi_tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)  #Tracks shape: torch.Size([2,1,64000]), no batch dim here
       

        #save to out, with tempo and step vectors info, and listen
        #sequencer.save_multi_tracks(multi_tracks, samples, step_vectors, tempo, OUT_DIR, save_sum_track_only = False, stereo=True)

        break
    


#to check: sounds like some one-shot start quite late in its attack phase
"""
sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)

class KonSequencerModel(pl.LightningModule):
    def __init__(self, sample_rate, num_tracks,num_steps=8 ):
        super(KonSequencerModel, self).__init__()
        self.sample_rate = sample_rate
        self.num_steps = num_steps
        self.conv1 = nn.Conv1d(1, 4, kernel_size=9, stride=3, padding=1)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=9, stride=3, padding=1)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=9, stride=3, padding=1)
        self.fc1 = nn.Linear(18944, 64)
        self.fc2 = nn.Linear(64, 1+num_tracks*num_steps)  # 1 for tempo + 8 for step vector

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def _get_preds_loss_accuracy(self, batch):
        #convenience function since train/valid/test steps are similar
        samples, step_vectors, tempos, one_shot_names = batch
        # Combine tempos and step_vectors to form the target
        target_step_vectors = step_vectors.view(step_vectors.shape[0], -1)
        #print("Target step vectors shape after torch.cat along dim 1:", target_step_vectors.shape) #torch.Size([16, 16])
        #tempo: batch, 1
        #target: batch, 1 + num_tracks*num_steps
        targets = torch.cat([tempos, target_step_vectors], dim=1).float() 

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
        #print("Synthesized loops batch shape:", synthesized_loops_batch.shape) #shape: torch.Size([16, 1, 64000])


        # Predict tempo and step vector
        y_pred = self(synthesized_loops_batch)

        # Loss is computed separately for tempo and step vector
        tempo_loss = nn.functional.mse_loss(y_pred[:, 0], targets[:, 0])
        step_vector_loss = nn.functional.binary_cross_entropy_with_logits(y_pred[:, 1:], targets[:, 1:])

        loss = tempo_loss + step_vector_loss

        return y_pred, tempo_loss, step_vector_loss, loss

    def training_step(self, batch, batch_idx):
        y_pred, tempo_loss, step_vector_loss, loss = self._get_preds_loss_accuracy(batch)
        self.log('train_tempo_loss', tempo_loss)
        self.log('train_step_vector_loss', step_vector_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, tempo_loss, step_vector_loss, loss = self._get_preds_loss_accuracy(batch)
        self.log('val_tempo_loss', tempo_loss)
        self.log('val_step_vector_loss', step_vector_loss)
        self.log('val_total_loss', loss)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = KonSequencerModel(sample_rate=SAMPLE_RATE, num_tracks = NUM_TRACKS, num_steps=NUM_STEPS)
wandb_logger = WandbLogger(log_model="all")
wandb_logger.watch(model,log_freq=MODEL_WATCH_FREQ)
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger,log_every_n_steps=TRAINER_LOG_EVERY_N_STEPS)
trainer.fit(model, train_data_loader, val_data_loader)

#train/val/test split: test: manual labelled loops!
#split one-shot samples, fixed sequencer step vectors for validation sets!
#train a simple model using w and b.
#place holder step vectors for absent tracks






