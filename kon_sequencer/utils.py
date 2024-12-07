import os
import torchaudio
import json

def save_multi_tracks(sum_track, multi_tracks, one_shot_samples, tempo, output_dir, sample_rate, instru_names = ["kick", "snare", "hihats"], save_multitracks = False, stereo = False):
    #Tracks shape: torch.Size([2,1,64000])  #no batch dim here

    if save_multitracks:
        for idx, one_track in enumerate(multi_tracks):
            track_save_path = os.path.join(output_dir, f"{instru_names[idx]}.wav")
            torchaudio.save(track_save_path, one_track, sample_rate=sample_rate)

            oneshot_save_path = os.path.join(output_dir, f"one_shot_{instru_names[idx]}.wav")
            torchaudio.save(oneshot_save_path, one_shot_samples[idx], sample_rate=sample_rate)
    
    #sum_track is mono by default
    sum_track_save_path = os.path.join(output_dir, f"sum_track_bpm{tempo}.wav")
    if stereo:
        sum_track = sum_track.repeat(2, 1) 
    torchaudio.save(sum_track_save_path, sum_track, sample_rate= sample_rate)

def log_info_to_json(output_dir, step_vectors,sample_names, tempo):
    meta_data = {
        "kick_step_vector": ' '.join(map(lambda x: str(int(x)), step_vectors[0].tolist())),
        "snare_step_vector": ' '.join(map(lambda x: str(int(x)), step_vectors[1].tolist())),
        "hh_step_vector": ' '.join(map(lambda x: str(int(x)), step_vectors[2].tolist())),
        "tempo": tempo,
        "kick_sample_name": sample_names[0],
        "snare_sample_name": sample_names[1],
        "hh_sample_name": sample_names[2]
    }
    json_file_path = os.path.join(output_dir, f"seq_params.json")
    with open(json_file_path, 'w') as f:
        json.dump(meta_data, f)








# You can now process these tensors using PyTorch


#one track prediction first: 
#one track prediction first: 

#lmdb converter: convert e.g. a kick folder into a train/val lmdb ! no need bc i will read one-shot dataset into memory for speed 

#tempo generator: randomly generate a tempo within a range of 60-200 bpm, or over a distribution

#one-shot sample fetcher: randomly generate one index and fetch one one-shot sample from the train/val lmdb

#one step vector generator: randomly generate one step vector,(the same for KK step vector, SN step cector, BD step vector)

#audio render: input KK step vector, SN step cector, BD step vector, N rounds, generate N random loops

#one training data: [a KK step vector, a SN step cector, a BD step vector, a loop]

#Method 1: complete randomness
#for each batch, (generate a random tempo, a set of step vectors, a set of kk, sn, hh one-shot samples which makes a different loop every time) x batch_size

#Method 2: sounds varying: semi-fixed step vectors, random one-shot sample fetching
#for each batch, (generate a random tempo and a set of step vectors that remains the same for N rounds of random sample fetching, N loops) x (batch_size/N)
#less randomness, might be biased and resulted in incorrect gradient 

#Method 3: rhythm varying: semi-fixed one-shot sample fetching, random step vector generation 
#for each batch, (fetch a set of kk, sn, hh one-shot samples that remains the same for N rounds of random step vector generation and tempo generation, N loops) x (batch_size/N)
#less randomness, might be biased and resulted in incorrect gradient


#tempo: regressor loss
#step vector: for each step it is a binary classifier loss, in total there are 24 steps (3 tracks x 8 steps)