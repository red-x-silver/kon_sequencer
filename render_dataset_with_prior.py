import json
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from kon_sequencer.paths import *
from kon_sequencer.params import *
from kon_sequencer.data_modules import  GlobalTempoSampler, MultiTrackDataset, OneShotSamplesDataset

from kon_sequencer.sequencer import KonSequencer

from kon_sequencer.utils import save_multi_tracks, log_info_to_json

# Set seed for NumPy
np.random.seed(GLOBAL_SEED)

# Set seed for PyTorch
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)  # Set the seed for all GPUs

#Set seed for random
random.seed(GLOBAL_SEED)

NUM_OF_LOOPS_TO_GENERATE = NUM_OF_LOOPS_TEST_PRIOR
SAVE_DIR = TEST_PRIORSET_DIR


tempo_sampler = GlobalTempoSampler(TEMPO_LOW, TEMPO_HIGH)
sequencer = KonSequencer(num_tracks=NUM_TRACKS, num_steps=NUM_STEPS, steps_per_beat=STEPS_PER_BEAT, sample_rate=SAMPLE_RATE, loop_length = TARGET_LOOP_LENGTH, left_safety_padding = LEFT_PADDING,right_safety_padding = RIGHT_PADDING)


kk_one_shots_dataset_val = OneShotSamplesDataset(KK_TEST_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
sn_one_shots_dataset_val = OneShotSamplesDataset(SN_TEST_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
hh_one_shots_dataset_val = OneShotSamplesDataset(HH_TEST_DIR,make_mono=MONO, unifyLenth=UNIFYSAMPLELEN, targetLength=ONE_SHOT_SAMPLE_LENGTH, ext = ["wav"]) # Replace with actual paths to your one-shot samples
kk_snare_hh_val = MultiTrackDataset(tempo_sampler = tempo_sampler, oneShotSamplesDataset_Lists = [kk_one_shots_dataset_val, sn_one_shots_dataset_val, hh_one_shots_dataset_val], num_steps=NUM_STEPS, num_tracks = NUM_TRACKS,dataset_size=TRAIN_DATA_SIZE) 
#val data loader
val_data_loader = DataLoader(kk_snare_hh_val, batch_size=1, shuffle=True)

data_iter = iter(val_data_loader)

def load_step_vectors(file_path):
    """
    Reads a JSON file containing step vectors and returns a list of vectors.

    :param file_path: Path to the JSON file.
    :return: List of 8-step sequencer vectors.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extracting and returning the step vectors
    return data.get("step_vectors", [])


kick_step_vectors = load_step_vectors(KK_PRIOR_STEP_VECTORS)
snare_step_vectors = load_step_vectors(SN_PRIOR_STEP_VECTORS)
hihats_step_vectors = load_step_vectors(HH_PRIOR_STEP_VECTORS)

def random_generate_step_vectors_from_prior(kick_step_vectors, snare_step_vectors, hihats_step_vectors):
    """
    Randomly selects step vectors from the prior set for each instrument.

    :return: A tuple containing the step vectors for each instrument.
    """
    kick_vector = torch.Tensor(random.choice(kick_step_vectors))
    snare_vector = torch.Tensor(random.choice(snare_step_vectors))
    hihats_vector = torch.Tensor(random.choice(hihats_step_vectors))
    step_vectors = (kick_vector, snare_vector, hihats_vector)
    return torch.stack(step_vectors, dim=0)



def generate_loops_with_prior_rhythms_and_save(data_iter, num_of_loops, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_of_loops):
        samples, random_step_vectors, tempos, sample_names = next(data_iter)
        #print(sample_names)
        #print("Samples shape of one batch:", samples.shape) #torch.Size([2, 1, 12800])
        #print("Step vectors shape of one batch:", step_vectors.shape) #torch.Size([2, 8])
        
        #grab step_vectors from the prior set
        step_vectors = random_generate_step_vectors_from_prior(kick_step_vectors, snare_step_vectors, hihats_step_vectors)
       
        for samples, step_vectors, tempo in zip(samples, step_vectors.unsqueeze(0), tempos):      
            tracks = sequencer.render_multi_tracks(samples, step_vectors, tempo)
            sum_track = torch.sum(tracks/len(tracks), dim=0)
            print(f"sum_track shape: {sum_track.shape}") #shape: torch.Size([1, 64000])
            one_loop_folder_path = os.path.join(output_dir, f"loop_{i}")
            os.makedirs(one_loop_folder_path, exist_ok=True)
            save_multi_tracks(sum_track=sum_track, multi_tracks = tracks, one_shot_samples = samples, tempo = tempo.item(), output_dir = one_loop_folder_path, sample_rate = SAMPLE_RATE, save_multitracks=True)
            log_info_to_json(output_dir = one_loop_folder_path, step_vectors = step_vectors, tempo = tempo.item(), sample_names = sample_names)


        #save to out, with tempo and step vectors info, and listen
        #sequencer.save_multi_tracks(multi_tracks, samples, step_vectors, tempo, OUT_DIR, save_sum_track_only = False, stereo=True)


if __name__ == "__main__":
    generate_loops_with_prior_rhythms_and_save(data_iter, NUM_OF_LOOPS_TO_GENERATE, SAVE_DIR)
    print("Done")
