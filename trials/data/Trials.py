
import time
import os
import pandas as pd
from data.aq_raw import EEG

# EPOC+ sensor order (14 bit channels)
SENSOR_ORDER = [
"COUNTER", 'F3',' FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8','F8', 'AF4', 'FC6', 'F4' 
]

def collect_stage_data(cyHeadset, stage_duration, filename):
    """Collects EEG data for a specified stage duration and saves it to a CSV."""
    data = []
    timestamps = []
    start_time = time.time()

    print(f"Collecting data for {stage_duration} seconds...")
    cyHeadset.clear_data()
    while time.time() - start_time < stage_duration:
        try:
            list_str = cyHeadset.get_data()
            if list_str is None:
                print("Received None from get_data(), skipping sample.")
                continue

            list_str = list_str.strip()
            if not list_str:
                print("Received empty string from get_data(), skipping sample.")
                continue

            list_values = list_str.split(',')

            if len(list_values) != len(SENSOR_ORDER):
                print(f"Incorrect number of values received: {len(list_values)}, expected {len(SENSOR_ORDER)}. Skipping sample.")
                print(f"Received data: {list_values}")
                continue

            counter = list_values[0]
            packet = list_values[1:]

            if packet:
                data.append([counter] + packet)
                timestamps.append(time.time() - start_time)
                print(f"Collected sample: {list_values}")
            else:
                print("Packet is empty, skipping sample.")

        except Exception as e:
            print(f"Error collecting data: {str(e)}")
            break

    if data:
        df = pd.DataFrame(data, columns=SENSOR_ORDER)
        df["Timestamp"] = timestamps
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        print(f"Collected {len(df)} samples")
    else:
        print("No data collected.")

def collect_word_data(cyHeadset, participant_folder, word_name):
    """Collects EEG data for each stage of a word."""
    word_folder = os.path.join(participant_folder, word_name)
    os.makedirs(word_folder, exist_ok=True)

    stages = ["vocal", "non-vocal", "motor", "motor+vocal", "motor+non-vocal"]
    stage_duration = 15  # seconds
    input("Press Enter to start Stage one")
    
    for i, stage_name in enumerate(stages):
        stage_folder = os.path.join(word_folder, stage_name)
        os.makedirs(stage_folder, exist_ok=True)

        collect_stage_data(cyHeadset, stage_duration, os.path.join(stage_folder, f"{stage_name}_record.csv"))

        if i < len(stages) - 1:
            input(f"Press Enter to start the next stage: {stages[i+1]}")

def collect_trial_data(words,name,date):
    """Collects EEG data for an entire trial."""

    trial_folder = f"trial_{name}_{date}"
    os.makedirs(trial_folder, exist_ok=True)

    cyHeadset = EEG()


    for word_name in words:
        collect_word_data(cyHeadset, trial_folder, word_name)
