from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from pathlib import Path
import os
import pandas as pd
import numpy as np
import json
import csv
from io import StringIO

def get_users(base_dir):
    """Get a list of existing participants from the Trials_data directory."""
    users = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and item.startswith('trial_'):
                # Extract participant name from directory name
                participant_name = item.replace('trial_', '')
                users.append(participant_name)
    
    return sorted(users)

def get_available_stages(user_folder, words):
    """Get a list of stages that have data for the given user and words."""
    available_stages = set()
    
    # Check each word directory
    for word in words:
        word_path = os.path.join(user_folder, word)
        if os.path.isdir(word_path):
            # Check each stage directory
            for stage_dir in os.listdir(word_path):
                stage_path = os.path.join(word_path, stage_dir)
                if os.path.isdir(stage_path):
                    # Only include stages with EEG data
                    eeg_files = [f for f in os.listdir(stage_path) if f.endswith('.csv') and 'eeg' in f.lower()]
                    if eeg_files:
                        available_stages.add(stage_dir)
    
    return sorted(list(available_stages))

def eeg_view(request):
    """Main view for the EEG visualization dashboard."""
    users = get_users(settings.TRIAL_DIR)
    selected_user = request.GET.get("user", "")
    
    words = []
    available_stages = []
    
    if selected_user:
        # Get available words and stages for the selected user
        user_folder = f"{settings.TRIAL_DIR}/trial_{selected_user}"
        
        if os.path.exists(user_folder):
            # Get words
            words = [d for d in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, d))]
            
            # Get available stages that have data
            available_stages = get_available_stages(user_folder, words)
    
    return render(request, 'plot.html', {
        "users": users, 
        "selected_user": selected_user, 
        "words": sorted(words),
        "available_stages": available_stages
    })

def get_eeg_data_api(request):
    """API endpoint to get EEG data for visualization."""
    user = request.GET.get("user", "")
    word = request.GET.get("word", "all")
    stage = request.GET.get("stage", "all")
    attempt = request.GET.get("attempt", "all")  # Add attempt parameter
    
    if not user:
        return JsonResponse({"error": "No user selected", "success": False}, status=400)
    
    user_folder = f"{settings.TRIAL_DIR}/trial_{user}"
    
    if not os.path.exists(user_folder):
        return JsonResponse({"error": "User not found", "success": False}, status=404)
    
    try:
        # Build a dictionary to hold channel data and events
        channel_data = {}
        events = []
        total_samples = 0
        
        # Function to process a single CSV file
        def process_eeg_csv(file_path, word_name=None, attempt_num=None):
            nonlocal channel_data, events, total_samples
            
            if not os.path.exists(file_path):
                return
            
            # Check if we need to filter by attempt
            if attempt != "all" and attempt_num and attempt_num != attempt:
                return
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                if "Timestamp" not in df.columns:
                    return
                
                # Check for potential word-specific event columns
                event_columns = [col for col in df.columns if col.endswith('_event')]
                
                # Extract timestamps
                timestamps = df["Timestamp"].values
                
                # Process each channel
                eeg_channels = [col for col in df.columns if col not in ["Timestamp", "COUNTER", "participant_id", "word", "stage", "attempt"] and not col.endswith('_event')]
                
                for channel in eeg_channels:
                    if channel not in channel_data:
                        channel_data[channel] = {"times": [], "values": []}
                    
                    # Add data points to the channel
                    channel_data[channel]["times"].extend(timestamps.tolist())
                    channel_data[channel]["values"].extend(df[channel].values.tolist())
                
                # Process events if any
                for event_col in event_columns:
                    word_name_from_col = event_col.replace('_event', '')
                    
                    # Get row indices where the event is True
                    event_rows = df[df[event_col] == True].index.tolist()
                    
                    # Detect transitions (speech onset)
                    for i in range(1, len(df)):
                        if i in event_rows and i-1 not in event_rows:
                            # This is a transition from False to True - speech onset
                            events.append({
                                "time": df.loc[i, "Timestamp"],
                                "label": f"{word_name_from_col} onset",
                                "type": "onset",
                                "attempt": attempt_num
                            })
                        elif i not in event_rows and i-1 in event_rows:
                            # This is a transition from True to False - speech offset
                            events.append({
                                "time": df.loc[i-1, "Timestamp"],
                                "label": f"{word_name_from_col} offset",
                                "type": "offset",
                                "attempt": attempt_num
                            })
                
                # Update total sample count
                total_samples += len(df)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Determine which files to process based on word, stage, and attempt filters
        if word == "all" and stage == "all":
            # Process all files for the user
            for word_dir in os.listdir(user_folder):
                word_path = os.path.join(user_folder, word_dir)
                if os.path.isdir(word_path):
                    for stage_dir in os.listdir(word_path):
                        stage_path = os.path.join(word_path, stage_dir)
                        if os.path.isdir(stage_path):
                            for file in os.listdir(stage_path):
                                if file.endswith(".csv") and "eeg" in file.lower():
                                    # Extract attempt number from filename
                                    attempt_num = None
                                    if "_attempt_" in file:
                                        attempt_num = file.split("_attempt_")[1].split(".")[0]
                                    
                                    file_path = os.path.join(stage_path, file)
                                    process_eeg_csv(file_path, word_dir, attempt_num)
        
        elif word != "all" and stage == "all":
            # Process all files for the specific word
            word_path = os.path.join(user_folder, word)
            if os.path.exists(word_path) and os.path.isdir(word_path):
                for stage_dir in os.listdir(word_path):
                    stage_path = os.path.join(word_path, stage_dir)
                    if os.path.isdir(stage_path):
                        for file in os.listdir(stage_path):
                            if file.endswith(".csv") and "eeg" in file.lower():
                                # Extract attempt number from filename
                                attempt_num = None
                                if "_attempt_" in file:
                                    attempt_num = file.split("_attempt_")[1].split(".")[0]
                                
                                file_path = os.path.join(stage_path, file)
                                process_eeg_csv(file_path, word, attempt_num)
        
        elif word == "all" and stage != "all":
            # Process all files for the specific stage
            for word_dir in os.listdir(user_folder):
                word_path = os.path.join(user_folder, word_dir)
                if os.path.isdir(word_path):
                    stage_path = os.path.join(word_path, stage)
                    if os.path.exists(stage_path) and os.path.isdir(stage_path):
                        for file in os.listdir(stage_path):
                            if file.endswith(".csv") and "eeg" in file.lower():
                                # Extract attempt number from filename
                                attempt_num = None
                                if "_attempt_" in file:
                                    attempt_num = file.split("_attempt_")[1].split(".")[0]
                                
                                file_path = os.path.join(stage_path, file)
                                process_eeg_csv(file_path, word_dir, attempt_num)
        
        else:
            # Process files for the specific word and stage
            word_path = os.path.join(user_folder, word)
            if os.path.exists(word_path) and os.path.isdir(word_path):
                stage_path = os.path.join(word_path, stage)
                if os.path.exists(stage_path) and os.path.isdir(stage_path):
                    for file in os.listdir(stage_path):
                        if file.endswith(".csv") and "eeg" in file.lower():
                            # Extract attempt number from filename
                            attempt_num = None
                            if "_attempt_" in file:
                                attempt_num = file.split("_attempt_")[1].split(".")[0]
                            
                            file_path = os.path.join(stage_path, file)
                            process_eeg_csv(file_path, word, attempt_num)
        
        # Sort each channel's data by time
        for channel in channel_data:
            # Convert to numpy arrays for easier manipulation
            times = np.array(channel_data[channel]["times"])
            values = np.array(channel_data[channel]["values"])
            
            # Sort by time
            sort_indices = np.argsort(times)
            times = times[sort_indices]
            values = values[sort_indices]
            
            # Update the channel data
            channel_data[channel]["times"] = times.tolist()
            channel_data[channel]["values"] = values.tolist()
        
        # Sort events by time
        events.sort(key=lambda x: x["time"])
        
        return JsonResponse({
            "channels": channel_data,
            "events": events,
            "totalSamples": total_samples,
            "attempt": attempt,
            "success": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e), "success": False}, status=500)
        
def export_csv_api(request):
    """API endpoint to export EEG data as CSV."""
    user = request.GET.get("user", "")
    word = request.GET.get("word", "all")
    stage = request.GET.get("stage", "all")
    attempt = request.GET.get("attempt", "all")  # Add attempt parameter
    
    if not user:
        return HttpResponse("No user selected", status=400)
    
    user_folder = f"{settings.TRIAL_DIR}/trial_{user}"
    
    if not os.path.exists(user_folder):
        return HttpResponse("User not found", status=404)
    
    try:
        # Create a CSV string buffer
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Write header
        header = ["Timestamp", "Channel", "Value", "Word", "Stage", "Attempt", "Event"]
        writer.writerow(header)
        
        # Function to process a single CSV file
        def process_eeg_csv_for_export(file_path, word_name, stage_name, attempt_num):
            if not os.path.exists(file_path):
                return
            
            # Check if we need to filter by attempt
            if attempt != "all" and attempt_num and attempt_num != attempt:
                return
                
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                if "Timestamp" not in df.columns:
                    return
                
                # Check for potential word-specific event columns
                event_columns = [col for col in df.columns if col.endswith('_event')]
                
                # Extract timestamps
                timestamps = df["Timestamp"].values
                
                # Process each channel
                eeg_channels = [col for col in df.columns if col not in ["Timestamp", "COUNTER", "participant_id", "word", "stage", "attempt"] and not col.endswith('_event')]
                
                for channel in eeg_channels:
                    for i in range(len(df)):
                        # Check if any event is active for this row
                        event_active = any(df.loc[i, ec] for ec in event_columns if ec in df.columns)
                        
                        # Write row for each channel
                        writer.writerow([
                            df.loc[i, "Timestamp"],
                            channel,
                            df.loc[i, channel],
                            word_name,
                            stage_name,
                            attempt_num,
                            "True" if event_active else "False"
                        ])
                        
            except Exception as e:
                print(f"Error processing {file_path} for export: {e}")
        
        # Determine which files to process based on word, stage, and attempt filters
        if word == "all" and stage == "all":
            # Process all files for the user
            for word_dir in os.listdir(user_folder):
                word_path = os.path.join(user_folder, word_dir)
                if os.path.isdir(word_path):
                    for stage_dir in os.listdir(word_path):
                        stage_path = os.path.join(word_path, stage_dir)
                        if os.path.isdir(stage_path):
                            for file in os.listdir(stage_path):
                                if file.endswith(".csv") and "eeg" in file.lower():
                                    # Extract attempt number from filename if possible
                                    attempt_num = "1"
                                    if "_attempt_" in file:
                                        attempt_num = file.split("_attempt_")[1].split(".")[0]
                                    
                                    file_path = os.path.join(stage_path, file)
                                    process_eeg_csv_for_export(file_path, word_dir, stage_dir, attempt_num)
        
        elif word != "all" and stage == "all":
            # Process all files for the specific word
            word_path = os.path.join(user_folder, word)
            if os.path.exists(word_path) and os.path.isdir(word_path):
                for stage_dir in os.listdir(word_path):
                    stage_path = os.path.join(word_path, stage_dir)
                    if os.path.isdir(stage_path):
                        for file in os.listdir(stage_path):
                            if file.endswith(".csv") and "eeg" in file.lower():
                                # Extract attempt number from filename
                                attempt_num = "1"
                                if "_attempt_" in file:
                                    attempt_num = file.split("_attempt_")[1].split(".")[0]
                                
                                file_path = os.path.join(stage_path, file)
                                process_eeg_csv_for_export(file_path, word, stage_dir, attempt_num)
        
        elif word == "all" and stage != "all":
            # Process all files for the specific stage
            for word_dir in os.listdir(user_folder):
                word_path = os.path.join(user_folder, word_dir)
                if os.path.isdir(word_path):
                    stage_path = os.path.join(word_path, stage)
                    if os.path.exists(stage_path) and os.path.isdir(stage_path):
                        for file in os.listdir(stage_path):
                            if file.endswith(".csv") and "eeg" in file.lower():
                                # Extract attempt number from filename
                                attempt_num = "1"
                                if "_attempt_" in file:
                                    attempt_num = file.split("_attempt_")[1].split(".")[0]
                                
                                file_path = os.path.join(stage_path, file)
                                process_eeg_csv_for_export(file_path, word_dir, stage, attempt_num)
        
        else:
            # Process files for the specific word and stage
            word_path = os.path.join(user_folder, word)
            if os.path.exists(word_path) and os.path.isdir(word_path):
                stage_path = os.path.join(word_path, stage)
                if os.path.exists(stage_path) and os.path.isdir(stage_path):
                    for file in os.listdir(stage_path):
                        if file.endswith(".csv") and "eeg" in file.lower():
                            # Extract attempt number from filename
                            attempt_num = "1"
                            if "_attempt_" in file:
                                attempt_num = file.split("_attempt_")[1].split(".")[0]
                            
                            file_path = os.path.join(stage_path, file)
                            process_eeg_csv_for_export(file_path, word, stage, attempt_num)
        
        # Prepare the response
        filename = f"eeg_data_{user}_{word}_{stage}"
        if attempt != "all":
            filename += f"_attempt_{attempt}"
        filename += ".csv"
        
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HttpResponse(f"Error exporting data: {str(e)}", status=500)