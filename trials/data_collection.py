import time
import os
import pandas as pd
from .data.aq_raw import EEG

# EPOC+ sensor order (14 bit channels)
SENSOR_ORDER = [
"COUNTER", 'F3',' FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8','F8', 'AF4', 'FC6', 'F4'
]
cyHeadset = EEG()
def collect_stage_data(stage_duration, filename):
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
        return True # data was collected
    else:
        print("No data collected.")
        return False # no data was collected