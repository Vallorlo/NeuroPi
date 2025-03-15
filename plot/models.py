from django.db import models
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pathlib import Path  


def get_users(base_dir):
    users = []
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path) and folder.startswith("trial_"):
            user_name = folder.replace("trial_", "")  
            users.append(user_name)
    print(users)
    return users

def process_eeg_data(user_folder):
    plots = {}

    for word in os.listdir(user_folder):
        word_path = os.path.join(user_folder, word)

        if os.path.isdir(word_path): 
            plots[word] = {} 

            for stage in os.listdir(word_path):
                stage_path = os.path.join(word_path, stage)

                if os.path.isdir(stage_path):  
                    plots[word][stage] = [] 

                    for file in os.listdir(stage_path):
                        if file.endswith(".csv"):
                            file_path = os.path.join(stage_path, file)

                            df = pd.read_csv(file_path)

                            if "Timestamp" not in df.columns:
                                continue  

                            timestamps = df["Timestamp"]
                            eeg_channels = df.columns[1:-1]  

                            start_time = timestamps.min()
                            end_time = timestamps.max()
                            time_window = 5
                            num_intervals = int(np.ceil((end_time - start_time) / time_window))

                            for channel in eeg_channels:
                                for i in range(min(num_intervals, 3)):  # 3 images per stage
                                    fig, ax = plt.subplots(figsize=(12, 6))

                                    t_start = start_time + i * time_window
                                    t_end = t_start + time_window

                                    segment = df[(timestamps >= t_start) & (timestamps < t_end)]

                                    ax.plot(segment["Timestamp"], segment[channel], label=channel, alpha=0.7)

                                    ax.set_xlabel("Time (s)")
                                    ax.set_ylabel("EEG Signal")
                                    ax.set_title(f"{word} - {stage} - {channel} ({t_start:.2f}s to {t_end:.2f}s)")
                                    ax.legend(loc="upper right", fontsize="small", ncol=3)
                                    ax.grid(True)

                                    buf = BytesIO()
                                    plt.savefig(buf, format="png")
                                    buf.seek(0)
                                    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                                    plt.close(fig)

                                    plots[word][stage].append(plot_base64)

    return plots
