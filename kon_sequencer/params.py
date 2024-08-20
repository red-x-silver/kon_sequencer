#sample rate from the one-shot datasets, will also be the sample rate for synthesized loops
SAMPLE_RATE = 16000
print(f"specified universal sample rate for both one-shot samples and synthesized loops: {SAMPLE_RATE}")

# DataModule
MONO = True
UNIFYSAMPLELEN = True
ONE_SHOT_SAMPLE_LENGTH_SECS = 0.8 #for better computing speed in the trial experiment
print(f"specified universal one-shot sample length (in seconds): {ONE_SHOT_SAMPLE_LENGTH_SECS}")


ONE_SHOT_SAMPLE_LENGTH = int(SAMPLE_RATE * ONE_SHOT_SAMPLE_LENGTH_SECS)  
print(f"calculated universal one-shot sample length (in sample numbers): {ONE_SHOT_SAMPLE_LENGTH}")

#Global tempo range
TEMPO_LOW = 60
TEMPO_HIGH = 200

#sequencer
NUM_TRACKS = 2
print(f"specified number of tracks of the sequencer: {NUM_TRACKS}")


NUM_STEPS = 8
STEPS_PER_BEAT = 2
assert STEPS_PER_BEAT == NUM_STEPS // 4

MAX_SAMPLES_PER_BEAT = int(SAMPLE_RATE * 60 // TEMPO_LOW) # tempo/60 is beats per second
MAX_SAMPLES_PER_STEP = int(MAX_SAMPLES_PER_BEAT // STEPS_PER_BEAT) #under slowest tempo, number of samples per step
print(f"calculated maximum samples per step: {MAX_SAMPLES_PER_STEP}")

TARGET_LOOP_LENGTH_SECS = 4
TARGET_LOOP_LENGTH = int(SAMPLE_RATE * TARGET_LOOP_LENGTH_SECS)

LEFT_PADDING = int(ONE_SHOT_SAMPLE_LENGTH - 1)
RIGHT_PADDING = int(ONE_SHOT_SAMPLE_LENGTH - MAX_SAMPLES_PER_STEP) #to ensure for the slowest tempo, there are enough 0's at the last step to be convoluted with one one-shot sample.





#training
BATCH_SIZE = 16


