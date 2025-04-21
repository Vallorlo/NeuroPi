import traceback
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .forms import ProcessingForm
import os
import sys
import io
import json
import zipfile
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import tempfile
from .process_eeg import (
    process_trial_data, 
    get_files_to_process,
    detect_speech_timestamps,
    preprocess_audio,
    process_file_with_segments,
    get_media_directory,
    clean_media_files
)
from sklearn.model_selection import train_test_split
import re
import datetime

def process_manual_review(request):
    """View for manually reviewing and adjusting speech detection results."""
    # Redirect stdout to capture debugging output
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    try:
        # Check if we have processing parameters in session
        if 'processing_params' not in request.session or 'files_to_process' not in request.session:
            return redirect('process_data')
        
        # Get processing parameters from session
        params = request.session['processing_params']
        files_to_process = json.loads(request.session['files_to_process'])
        
        # Get current file index from query parameter, default to 0
        file_index = int(request.GET.get('file', 0))
        
        # If file index is out of range, processing is complete
        if file_index >= len(files_to_process):
            # Get output directory from params
            output_dir = params.get('output_dir')
            
            # Prepare result stats and links for the complete page
            stats = {
                'trials_processed': len(files_to_process),
                'total_participants': len(set(f['participant'] for f in files_to_process if 'participant' in f)),
                'total_unique_words': len(set(f['word'] for f in files_to_process if 'word' in f)),
                'stages_processed': len(set(f['stage'] for f in files_to_process if 'stage' in f)),
                'attempts_processed': len(files_to_process),
                'visualizations_created': len([f for f in files_to_process if 'output_viz' in f]),
                'participant_list': list(set(f['participant'] for f in files_to_process if 'participant' in f)),
                'words_list': list(set(f['word'] for f in files_to_process if 'word' in f)),
                'audio_processed': len(files_to_process),
                'timestamp_processed': 0,  # We don't track this separately in manual mode
                'dataset_rows': 0
            }
            
            dataset_links = []
            
            try:
                # Check if we should generate a dataset
                if params.get('generate_dataset', False):
                    # Collect all processed files
                    processed_files = []
                    for file_data in files_to_process:
                        if file_data.get('processed') and 'output_path' in file_data:
                            processed_files.append(file_data['output_path'])
                    
                    if processed_files:
                        # Read and combine all processed files
                        all_data = []
                        for file_path in processed_files:
                            try:
                                df = pd.read_csv(file_path)
                                
                                # Extract metadata from file path
                                path_parts = os.path.normpath(file_path).split(os.sep)
                                participant = None
                                word = None
                                stage = None
                                attempt = None
                                
                                for i, part in enumerate(path_parts):
                                    if part.startswith('trial_') and i+1 < len(path_parts):
                                        participant = part.replace('trial_', '')
                                        if i+1 < len(path_parts):
                                            word = path_parts[i+1]
                                        if i+2 < len(path_parts) and 'stage' in path_parts[i+2].lower():
                                            stage_part = path_parts[i+2]
                                            stage_match = re.search(r'\d+', stage_part)
                                            if stage_match:
                                                stage = stage_match.group(0)
                                        break
                                
                                # Extract attempt from filename
                                filename = os.path.basename(file_path)
                                attempt_match = re.search(r'attempt[_]?(\d+)', filename)
                                if attempt_match:
                                    attempt = attempt_match.group(1)
                                
                                # Add metadata if not already in the dataframe
                                if participant and 'participant_id' not in df.columns:
                                    df['participant_id'] = participant
                                if word and 'word' not in df.columns:
                                    df['word'] = word
                                if stage and 'stage' not in df.columns:
                                    df['stage'] = stage
                                if attempt and 'attempt' not in df.columns:
                                    df['attempt'] = attempt
                                
                                all_data.append(df)
                            except Exception as e:
                                print(f"Error reading file {file_path}: {e}")
                        
                        if all_data:
                            # Combine all dataframes
                            combined_df = pd.concat(all_data, ignore_index=True)
                            
                            # Fix missing event columns - fill NaN values with False
                            all_columns = combined_df.columns.tolist()
                            event_columns = [col for col in all_columns if '_event' in col]
                            for col in event_columns:
                                if combined_df[col].isnull().any():
                                    combined_df[col] = combined_df[col].fillna(False)
                            
                            # Make sure the word_label column is filled correctly
                            if 'word_label' in combined_df.columns:
                                combined_df['word_label'] = combined_df['word_label'].fillna('sil')
                            
                            # Save combined dataset
                            combined_path = os.path.join(output_dir, "combined_eeg_dataset.csv")
                            combined_df.to_csv(combined_path, index=False)
                            print(f"Created combined dataset with {len(combined_df)} rows at {combined_path}")
                            
                            # Add to dataset links
                            dataset_links.append({
                                'name': 'Combined Dataset',
                                'path': combined_path,
                                'filename': f'{os.path.basename(output_dir)}/combined_eeg_dataset.csv',
                                'size': f"{os.path.getsize(combined_path) / (1024*1024):.2f} MB"
                            })
                            
                            stats['dataset_rows'] = len(combined_df)
                            
                            # Create train-test split if requested
                            if params.get('create_train_test', False):
                                try:
                                    # Get split parameters
                                    test_size = params.get('test_size', 0.2)
                                    random_state = params.get('random_state', 42)
                                    stratify_by_word = params.get('stratify_by_word', True)
                                    
                                    # Determine stratification
                                    stratify = None
                                    if stratify_by_word:
                                        if 'word_label' in combined_df.columns:
                                            # For word_label, we need to handle 'sil' differently
                                            # as it would dominate. Instead, use the 'word' column
                                            # which has the actual word for each row
                                            stratify = combined_df['word']
                                        elif 'word' in combined_df.columns:
                                            stratify = combined_df['word']
                                    
                                    # Create split
                                    train_df, test_df = train_test_split(
                                        combined_df, 
                                        test_size=test_size,
                                        random_state=random_state,
                                        stratify=stratify
                                    )
                                    
                                    # Save splits
                                    train_path = os.path.join(output_dir, "train_dataset.csv")
                                    test_path = os.path.join(output_dir, "test_dataset.csv")
                                    
                                    train_df.to_csv(train_path, index=False)
                                    test_df.to_csv(test_path, index=False)
                                    
                                    print(f"Created train dataset with {len(train_df)} rows")
                                    print(f"Created test dataset with {len(test_df)} rows")
                                    
                                    # Add to dataset links
                                    dataset_links.append({
                                        'name': 'Training Dataset',
                                        'path': train_path,
                                        'filename': f'{os.path.basename(output_dir)}/train_dataset.csv',
                                        'size': f"{os.path.getsize(train_path) / (1024*1024):.2f} MB"
                                    })
                                    
                                    dataset_links.append({
                                        'name': 'Testing Dataset',
                                        'path': test_path,
                                        'filename': f'{os.path.basename(output_dir)}/test_dataset.csv',
                                        'size': f"{os.path.getsize(test_path) / (1024*1024):.2f} MB"
                                    })
                                    
                                    # Update stats
                                    stats['train_rows'] = len(train_df)
                                    stats['test_rows'] = len(test_df)
                                
                                except Exception as e:
                                    print(f"Error creating train-test split: {e}")
                
            except Exception as e:
                print(f"Error creating combined dataset: {e}")
            
            # Remove parameters from session
            del request.session['processing_params']
            del request.session['files_to_process']
            
            # Create success message
            message = "Processing complete! "
            if params.get('generate_dataset', False):
                message += f"Combined dataset created with {stats.get('dataset_rows', 0)} rows. "
            if params.get('create_train_test', False) and 'train_rows' in stats:
                message += f"Train/test datasets created with {stats.get('train_rows', 0)}/{stats.get('test_rows', 0)} rows."
            
            # Render the process_complete template
            return render(request, 'preprocessor/process_complete.html', {
                'message': message,
                'stats': stats,
                'dataset_links': dataset_links,
                'output': output.getvalue()
            })
        
        # Get current file to process
        file_data = files_to_process[file_index]
        
        context = {
            'file_data': file_data,
            'file_index': file_index,
            'total_files': len(files_to_process),
            'next_file_index': file_index + 1,
            'enable_audio_playback': params.get('enable_audio_playback', True),
            'reprocess': params.get('reprocess', False),
            'output': '',
        }
        
        # Handle form submission for this file
        if request.method == 'POST':
            # Get segments from form data
            segments_json = request.POST.get('segments_json', '[]')
            try:
                segments = json.loads(segments_json)
                
                # Get output directory
                output_dir = params.get('output_dir')
                
                # Extract metadata from file_data
                participant = file_data['participant']
                word = file_data['word']
                stage = file_data['stage']
                attempt = file_data['attempt']
                
                # Process file with segments
                output_path, output_viz = process_file_with_segments(
                    file_data['eeg_file_path'],
                    segments,
                    file_data['word'],
                    output_dir=output_dir,
                    reprocess=params.get('reprocess', False),
                    participant=participant,
                    stage=stage,
                    attempt=attempt
                )
                
                # Update the file information in the session
                files_to_process[file_index]['processed'] = True
                files_to_process[file_index]['output_path'] = output_path
                files_to_process[file_index]['output_viz'] = output_viz
                request.session['files_to_process'] = json.dumps(files_to_process)
                
                # Redirect to next file
                return HttpResponseRedirect(f'/preprocessor/manual_review/?file={file_index + 1}')
                
            except Exception as e:
                context['error'] = f"Error processing segments: {e}"
                traceback.print_exc()
        
        # Initialize speech_segments as an empty list
        speech_segments = []
        
        # Get reprocess flag
        reprocess = params.get('reprocess', False)
        
        if file_data['audio_file_path']:
            # Use audio file for speech detection
            try:
                original_audio_path = file_data['audio_file_path']
                print(f"Original audio file: {original_audio_path}")
                
                # Extract metadata from file_data for centralized file handling
                participant = file_data['participant']
                word = file_data['word']
                stage = file_data['stage']
                attempt = file_data['attempt']
                
                # Get directory where audio file is located (should be stage folder)
                audio_dir = os.path.dirname(original_audio_path)
                
                # Process the audio file with the centralized file management
                processed_audio_path = preprocess_audio(
                    original_audio_path,
                    output_dir=params.get('output_dir'),
                    noise_reduction_strength=params.get('noise_reduction_strength', 0.5) if params.get('noise_reduction', True) else 0.0,
                    apply_highpass=params.get('audio_highpass', True),
                    highpass_cutoff=params.get('audio_highpass_cutoff', 150),
                    apply_lowpass=params.get('audio_lowpass', True),
                    lowpass_cutoff=params.get('audio_lowpass_cutoff', 5000),
                    reprocess=reprocess,
                    participant=participant,
                    word=word,
                    stage=stage,
                    attempt=attempt
                )
                
                print(f"Processed audio file: {processed_audio_path}")
                
                # Store the processed audio path in the session for this file_index
                request.session[f'audio_path_{file_index}'] = processed_audio_path
                
                # Extract just the filename to use in the URL
                processed_audio_filename = os.path.basename(processed_audio_path)
                
                # Generate a URL to directly serve this file
                audio_url = f"/preprocessor/audio/{file_index}/{processed_audio_filename}"
                context['processed_audio_url'] = audio_url
                context['audio_file_name'] = processed_audio_filename
                
                # Use the processed audio file for speech detection
                raw_speech_markers = detect_speech_timestamps(
                    processed_audio_path,
                    min_silence_len=params.get('min_silence', 300),
                    silence_thresh=params.get('silence_thresh', -40),
                    preprocess=False,  # Already preprocessed
                    noise_reduction_strength=0.0,
                    apply_highpass=False,
                    apply_lowpass=False,
                    reprocess=reprocess,
                    participant=participant,
                    word=word,
                    stage=stage,
                    attempt=attempt
                )
                
                # Ensure we have proper start/end pairs
                if raw_speech_markers and isinstance(raw_speech_markers, list):
                    if all(isinstance(marker, tuple) and len(marker) == 2 for marker in raw_speech_markers):
                        # Data is already in the correct format (start, end pairs)
                        speech_segments = raw_speech_markers
                    else:
                        # Convert flat list of timestamps to start/end pairs
                        pairs = []
                        for i in range(0, len(raw_speech_markers), 2):
                            if i + 1 < len(raw_speech_markers):
                                # A complete pair
                                pairs.append([raw_speech_markers[i], raw_speech_markers[i+1]])
                            else:
                                # Last element without a pair
                                pairs.append([raw_speech_markers[i], raw_speech_markers[i] + 0.5])
                        speech_segments = pairs
                        
                print(f"Processed speech segments: {speech_segments}")
                
            except Exception as e:
                print(f"Error processing audio: {e}")
                traceback.print_exc()
                context['audio_error'] = str(e)
                
        elif file_data['timestamp_file_path']:
            # Use timestamp file
            from .process_eeg import read_timestamp_file
            try:
                raw_timestamps = read_timestamp_file(
                    file_data['timestamp_file_path'],
                    padding=params.get('timestamp_padding', 0.25)
                )
                
                # Ensure timestamps are in the right format
                if isinstance(raw_timestamps, list):
                    if all(isinstance(ts, tuple) and len(ts) == 2 for ts in raw_timestamps):
                        # Already in start/end pair format
                        speech_segments = raw_timestamps
                    else:
                        # Convert to start/end pairs
                        pairs = []
                        padding = params.get('timestamp_padding', 0.25)
                        for timestamp in raw_timestamps:
                            if isinstance(timestamp, (int, float)):
                                start = max(0, timestamp - padding)
                                end = timestamp + padding
                                pairs.append([start, end])
                        speech_segments = pairs
                
            except Exception as e:
                print(f"Error reading timestamp file: {e}")
                traceback.print_exc()
                context['timestamp_error'] = str(e)
        
        # Convert speech_segments to proper format and ensure it's valid
        if speech_segments:
            # Make sure all segments are lists and not tuples (for JSON serialization)
            speech_segments = [[float(segment[0]), float(segment[1])] for segment in speech_segments]
        
        context['speech_segments'] = speech_segments
        
        # Load EEG data for visualization
        try:
            eeg_data = pd.read_csv(file_data['eeg_file_path'])
            context['eeg_data'] = {
                'timestamps': eeg_data['Timestamp'].tolist(),
                'duration': eeg_data['Timestamp'].max() - eeg_data['Timestamp'].min(),
                'num_samples': len(eeg_data),
                'sample_rate': 128  # Default sample rate for EEG data
            }
            
        except Exception as e:
            context['error'] = f"Error loading EEG data: {e}"
            traceback.print_exc()
        
        context['output'] = output.getvalue()
        
        return render(request, 'preprocessor/manual_review.html', context)
        
    finally:
        sys.stdout = old_stdout


def process_data_view(request):
    message = ''
    output = io.StringIO()  # Use StringIO to capture print output
    old_stdout = sys.stdout  # Store the original stdout
    sys.stdout = output     # Redirect stdout to our StringIO object

    # --- Set the default root directory here ---
    default_root_dir = os.path.join(settings.BASE_DIR, 'Trials_data')

    # --- Create the directory if it doesn't exist ---
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)
        
    # Get available participants and words for the filter UI
    available_participants = []
    available_words = []
    
    try:
        # Scan the trials directory to find available participants and words
        if os.path.exists(default_root_dir):
            # Only include actual participant directories (those starting with 'trial_')
            available_participants = [d.replace('trial_', '') for d in os.listdir(default_root_dir) 
                                     if os.path.isdir(os.path.join(default_root_dir, d)) 
                                     and d.startswith('trial_')]
            
            # Get words from the first participant (assuming similar structure for all)
            if available_participants:
                first_participant = os.path.join(default_root_dir, f'trial_{available_participants[0]}')
                available_words = [d for d in os.listdir(first_participant)
                                   if os.path.isdir(os.path.join(first_participant, d))]
    except Exception as e:
        print(f"Error scanning directories: {e}")

    # Create a dynamic choices list for form fields
    participant_choices = [(p, p) for p in available_participants]
    word_choices = [(w, w) for w in available_words]

    try:
        if request.method == 'POST':
            # Check if this is an AJAX request for participant words
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' and 'get_words' in request.POST:
                participant = request.POST.get('participant')
                participant_dir = os.path.join(default_root_dir, f'trial_{participant}')
                
                if os.path.exists(participant_dir):
                    words = [d for d in os.listdir(participant_dir)
                           if os.path.isdir(os.path.join(participant_dir, d))]
                    return JsonResponse({'words': words})
                return JsonResponse({'words': []})
            
            # Create form with custom choices
            form = ProcessingForm(request.POST, request.FILES)
            
            # Important: Dynamically update form field choices to match available options
            form.fields['selected_participants'].choices = participant_choices
            form.fields['selected_words'].choices = word_choices
            
            if form.is_valid():
                # Extract form data
                # Basic options
                verbose = form.cleaned_data['verbose']
                create_visualizations = form.cleaned_data['create_visualizations']
                generate_dataset = form.cleaned_data['generate_dataset']
                manual_processing = form.cleaned_data['manual_processing']
                enable_audio_playback = form.cleaned_data['enable_audio_playback']
                
                # NEW: Reprocessing option
                reprocess = form.cleaned_data['reprocess']
                
                # Speech detection parameters
                silence_thresh = form.cleaned_data['silence_thresh']
                min_silence = form.cleaned_data['min_silence']
                timestamp_padding = form.cleaned_data['timestamp_padding']
                
                # Advanced audio processing options
                noise_reduction = form.cleaned_data['noise_reduction']
                noise_reduction_strength = form.cleaned_data['noise_reduction_strength']
                audio_highpass = form.cleaned_data['audio_highpass']
                audio_highpass_cutoff = form.cleaned_data['audio_highpass_cutoff']
                audio_lowpass = form.cleaned_data['audio_lowpass']
                audio_lowpass_cutoff = form.cleaned_data['audio_lowpass_cutoff']
                
                zip_file = request.FILES.get('zip_file')
                
                # Train-test split options
                create_train_test = form.cleaned_data['create_train_test_split']
                test_size = form.cleaned_data['test_size'] if create_train_test else 0.2
                random_state = form.cleaned_data['random_state'] if create_train_test else 42
                stratify_by_word = form.cleaned_data['stratify_by_word'] if create_train_test else False
                
                # Participant and word filters
                selected_participants = form.cleaned_data['selected_participants']
                selected_words = form.cleaned_data['selected_words']

                # Get selected stages
                selected_stages = []
                if form.cleaned_data['stage_1']:
                    selected_stages.append(1)
                if form.cleaned_data['stage_2']:
                    selected_stages.append(2)
                if form.cleaned_data['stage_3']:
                    selected_stages.append(3)
                if form.cleaned_data['stage_4']:
                    selected_stages.append(4)
                if form.cleaned_data['stage_5']:
                    selected_stages.append(5)

                root_dir = default_root_dir  # Use the default directory

                if zip_file:
                    temp_dir = tempfile.mkdtemp()
                    fs = FileSystemStorage(location=temp_dir)
                    filename = fs.save(zip_file.name, zip_file)
                    zip_file_path = fs.path(filename)

                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    root_dir = temp_dir  # Override with temp dir if zip is uploaded

                # Create a timestamped output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir_name = f'processed_{timestamp}'
                output_dir = os.path.join(root_dir, output_dir_name)
                os.makedirs(output_dir, exist_ok=True)

                # If manual processing is enabled, we need to redirect to the manual review page
                if manual_processing:
                    # Get list of files to process
                    files_to_process = get_files_to_process(
                        root_dir=root_dir,
                        selected_participants=selected_participants,
                        selected_words=selected_words,
                        selected_stages=selected_stages
                    )
                    
                    if not files_to_process:
                        message = "No files found to process with the current selection."
                        return render(request, 'preprocessor/process_data.html', {
                            'form': form, 
                            'message': message, 
                            'output': output.getvalue(),
                            'available_participants': available_participants,
                            'available_words': available_words,
                            'has_processed_data': False,
                        })
                    
                    # Store processing parameters in session for the manual review page
                    request.session['processing_params'] = {
                        'root_dir': root_dir,
                        'output_dir': output_dir,
                        'timestamp': timestamp,
                        'silence_thresh': silence_thresh,
                        'min_silence': min_silence,
                        'timestamp_padding': timestamp_padding,
                        'noise_reduction': noise_reduction,
                        'noise_reduction_strength': noise_reduction_strength,
                        'audio_highpass': audio_highpass,
                        'audio_highpass_cutoff': audio_highpass_cutoff,
                        'audio_lowpass': audio_lowpass,
                        'audio_lowpass_cutoff': audio_lowpass_cutoff,
                        'enable_audio_playback': enable_audio_playback,
                        'generate_dataset': generate_dataset,
                        'create_train_test': create_train_test,
                        'test_size': test_size,
                        'random_state': random_state,
                        'stratify_by_word': stratify_by_word,
                        'selected_stages': selected_stages,
                        'reprocess': reprocess,  # Include reprocessing option
                    }
                    
                    # Serialize the files to process
                    request.session['files_to_process'] = json.dumps([{
                        'eeg_file_path': f["eeg_file_path"],
                        'audio_file_path': f["audio_file_path"] if f["audio_file_path"] else '',
                        'timestamp_file_path': f["timestamp_file_path"] if f["timestamp_file_path"] else '',
                        'participant': f["participant"],
                        'word': f["word"],
                        'stage': f["stage"],
                        'attempt': f["attempt"]
                    } for f in files_to_process])
                    
                    # Redirect to manual review page with file index 0
                    return redirect('process_manual_review')

                try:
                    # If reprocessing is enabled, clean media files first
                    if reprocess:
                        media_dir = get_media_directory(output_dir)
                        clean_media_files(media_dir)
                        print(f"Cleaned media directory for reprocessing: {media_dir}")
                        
                    stats = process_trial_data(
                        root_dir=root_dir,
                        output_dir=output_dir,
                        verbose=verbose,
                        create_visualizations=create_visualizations,
                        generate_dataset=generate_dataset,
                        timestamp_padding=timestamp_padding,
                        selected_stages=selected_stages,
                        selected_participants=selected_participants,
                        selected_words=selected_words,
                        create_train_test=create_train_test,
                        test_size=test_size,
                        random_state=random_state,
                        stratify_by_word=stratify_by_word,
                        preprocess_audio=True,
                        noise_reduction_strength=noise_reduction_strength if noise_reduction else 0.0,
                        silence_thresh=silence_thresh,
                        min_silence_len=min_silence,
                        reprocess=reprocess  # Pass the reprocessing flag
                    )
                    message = "Processing complete!"
                    
                    # If datasets were created, add links to download them
                    if generate_dataset:
                        dataset_links = []
                        combined_dataset_path = os.path.join(output_dir, "combined_eeg_dataset.csv")
                        if os.path.exists(combined_dataset_path):
                            dataset_links.append({
                                'name': 'Combined Dataset',
                                'path': combined_dataset_path,
                                'filename': f'{output_dir_name}/combined_eeg_dataset.csv',  # Include subdirectory
                                'size': f"{os.path.getsize(combined_dataset_path) / (1024*1024):.2f} MB"
                            })
                        
                        if create_train_test:
                            train_path = os.path.join(output_dir, "train_dataset.csv")
                            test_path = os.path.join(output_dir, "test_dataset.csv")
                            
                            if os.path.exists(train_path):
                                dataset_links.append({
                                    'name': 'Training Dataset',
                                    'path': train_path,
                                    'filename': f'{output_dir_name}/train_dataset.csv',  # Include subdirectory
                                    'size': f"{os.path.getsize(train_path) / (1024*1024):.2f} MB"
                                })
                            
                            if os.path.exists(test_path):
                                dataset_links.append({
                                    'name': 'Testing Dataset',
                                    'path': test_path,
                                    'filename': f'{output_dir_name}/test_dataset.csv',  # Include subdirectory
                                    'size': f"{os.path.getsize(test_path) / (1024*1024):.2f} MB"
                                })
                        message += " Generated datasets are available for download below."
                        
                        return render(request, 'preprocessor/process_complete.html', {
                            'message': message,
                            'dataset_links': dataset_links,
                            'stats': stats,
                            'output': output.getvalue(),
                            'output_dir': output_dir_name
                        })

                except Exception as e:
                    message = f"An error occurred: {e}"
                    print(f"Processing error: {e}")
                    import traceback
                    traceback.print_exc()  # Print the full traceback for debugging

                    # Return an error response with JavaScript to remove the overlay
                    response_html = f"""
                    <html>
                    <head>
                        <script>
                            // Remove the processing overlay first
                            function removeOverlay() {{
                                const overlay = document.getElementById('processing-overlay');
                                if (overlay) {{
                                    overlay.remove();
                                }}
                            }}
                            removeOverlay();
                            
                            // Then alert the user about the error
                            alert('Error during processing: {str(e)}');
                            
                            // Redirect back to the form page
                            window.location.href = window.location.href;
                        </script>
                    </head>
                    <body>
                        <p>Error processing request. Redirecting...</p>
                    </body>
                    </html>
                    """
                    return HttpResponse(response_html)

                if zip_file:
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        print(f"Temp directory '{temp_dir}' removed successfully.")
                    except OSError as e:
                         print(f"Error removing temp directory: {e}")
            else:
                print("Form is NOT valid")  # Debug: Form invalid
                message = "Invalid form inputs. Please check the error messages."
                print(form.errors)  # Print Errors
                
                # Add JavaScript to remove the processing overlay for invalid form submissions
                response_html = """
                <html>
                <head>
                    <script>
                        // Remove the processing overlay
                        function removeOverlay() {
                            const overlay = document.getElementById('processing-overlay');
                            if (overlay) {
                                overlay.remove();
                            }
                        }
                        removeOverlay();
                        
                        // Redirect back to the form page
                        window.location.href = window.location.href;
                    </script>
                </head>
                <body>
                    <p>Invalid form submission. Redirecting...</p>
                </body>
                </html>
                """
                return HttpResponse(response_html)
                
        else:
            # Create form with custom choices for initial page load
            form = ProcessingForm()
            form.fields['selected_participants'].choices = participant_choices
            form.fields['selected_words'].choices = word_choices

        # Check for existing processed data
        processed_dirs = [d for d in os.listdir(default_root_dir) 
                         if os.path.isdir(os.path.join(default_root_dir, d)) 
                         and d.startswith('processed_')]
        
        has_processed_data = len(processed_dirs) > 0
        latest_processed = max(processed_dirs, default=None) if processed_dirs else None

        return render(request, 'preprocessor/process_data.html', {
            'form': form, 
            'message': message, 
            'output': output.getvalue(),
            'available_participants': available_participants,
            'available_words': available_words,
            'has_processed_data': has_processed_data,
            'latest_processed': latest_processed
        })

    finally: # Very important to restore sys.stdout
        sys.stdout = old_stdout # Restore the original stdout



def download_dataset(request, filename):
    """View to handle dataset downloads"""
    # Handle paths that might include subdirectories
    path_parts = filename.split('/')
    
    if len(path_parts) > 1:
        # If the file is in a subdirectory
        subdir = path_parts[0]
        file_name = path_parts[1]
        file_path = os.path.join(settings.BASE_DIR, 'Trials_data', subdir, file_name)
    else:
        # Direct file in Trials_data
        file_path = os.path.join(settings.BASE_DIR, 'Trials_data', filename)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    
    # File not found, try searching
    trials_data_path = os.path.join(settings.BASE_DIR, 'Trials_data')
    for root, dirs, files in os.walk(trials_data_path):
        if os.path.basename(filename) in files:
            file_path = os.path.join(root, os.path.basename(filename))
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                return response
    
    return HttpResponse("File not found", status=404)



def serve_audio_file(request, filename, file_index=None):
    """View to serve audio files for playback in manual review mode"""
    # First check if we have a file_index parameter and a saved audio path in session
    if file_index is not None and f'audio_path_{file_index}' in request.session:
        # Get the full file path from session
        file_path = request.session[f'audio_path_{file_index}']
        print(f"Serving audio file from session for file_index {file_index}: {file_path}")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                content_type = 'audio/wav'
                response = HttpResponse(file_content, content_type=content_type)
                response['Content-Disposition'] = f'inline; filename="{os.path.basename(file_path)}"'
                return response
            except Exception as e:
                print(f"Error serving audio file from session: {e}")
                pass  # Continue to fallback methods
    
    # Look for the file in the centralized media directory
    media_dir = get_media_directory(os.path.join(settings.BASE_DIR, 'Trials_data'))
    processed_audio_dir = os.path.join(media_dir, 'processed_audio')
    if os.path.exists(processed_audio_dir):
        for root, dirs, files in os.walk(processed_audio_dir):
            if filename in files:
                file_path = os.path.join(root, filename)
                print(f"Serving audio file from media directory: {file_path}")
                
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    content_type = 'audio/wav'
                    response = HttpResponse(file_content, content_type=content_type)
                    response['Content-Disposition'] = f'inline; filename="{filename}"'
                    return response
                except Exception as e:
                    print(f"Error serving audio file from media directory: {e}")
                    break  # Continue to next method
    
    # If direct session path failed, check if we can find the file directly
    if not file_index:
        # Check in Trials_data and all subdirectories
        potential_paths = []
        trials_data_path = os.path.join(settings.BASE_DIR, 'Trials_data')
        for root, dirs, files in os.walk(trials_data_path):
            for file in files:
                if file == filename:
                    potential_paths.append(os.path.join(root, file))
                    print(f"Found audio file: {file}")
        
        # If found in Trials_data, serve the first one
        if potential_paths:
            file_path = potential_paths[0]
            print(f"Serving audio file from Trials_data: {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                content_type = 'audio/wav'
                response = HttpResponse(file_content, content_type=content_type)
                response['Content-Disposition'] = f'inline; filename="{os.path.basename(file_path)}"'
                return response
            except Exception as e:
                print(f"Error serving audio file from Trials_data: {e}")
                return HttpResponse(f"Error opening audio file: {str(e)}", status=500)
    
    # Additional diagnostic information
    print(f"Audio file not found: {filename}")
    if file_index:
        print(f"Requested for file_index: {file_index}")
    
    return HttpResponse("Audio file not found. This could be because the file wasn't generated correctly or the path is incorrect.", status=404)