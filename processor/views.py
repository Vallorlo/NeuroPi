# processor/views.py
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
from .process_eeg import process_trial_data  # Import directly!


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
            available_participants = [d for d in os.listdir(default_root_dir) 
                                     if os.path.isdir(os.path.join(default_root_dir, d))]
            
            # Get words from the first participant (assuming similar structure for all)
            if available_participants:
                first_participant = os.path.join(default_root_dir, available_participants[0])
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
                participant_dir = os.path.join(default_root_dir, participant)
                
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
                verbose = form.cleaned_data['verbose']
                create_visualizations = form.cleaned_data['create_visualizations']
                generate_dataset = form.cleaned_data['generate_dataset']
                silence_thresh = form.cleaned_data['silence_thresh']
                min_silence = form.cleaned_data['min_silence']
                timestamp_padding = form.cleaned_data['timestamp_padding']
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

                try:
                    stats = process_trial_data(
                        root_dir=root_dir,
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
                        stratify_by_word=stratify_by_word
                    )
                    message = "Processing complete!"
                    
                    # If datasets were created, add links to download them
                    if generate_dataset:
                        dataset_links = []
                        combined_dataset_path = os.path.join(root_dir, "combined_eeg_dataset.csv")
                        if os.path.exists(combined_dataset_path):
                            dataset_links.append({
                                'name': 'Combined Dataset',
                                'path': combined_dataset_path,
                                'filename': 'combined_eeg_dataset.csv',  # Add filename explicitly
                                'size': f"{os.path.getsize(combined_dataset_path) / (1024*1024):.2f} MB"
                            })
                        
                        if create_train_test:
                            train_path = os.path.join(root_dir, "train_dataset.csv")
                            test_path = os.path.join(root_dir, "test_dataset.csv")
                            
                            if os.path.exists(train_path):
                                dataset_links.append({
                                    'name': 'Training Dataset',
                                    'path': train_path,
                                    'filename': 'train_dataset.csv',  # Add filename explicitly
                                    'size': f"{os.path.getsize(train_path) / (1024*1024):.2f} MB"
                                })
                            
                            if os.path.exists(test_path):
                                dataset_links.append({
                                    'name': 'Testing Dataset',
                                    'path': test_path,
                                    'filename': 'test_dataset.csv',  # Add filename explicitly
                                    'size': f"{os.path.getsize(test_path) / (1024*1024):.2f} MB"
                                })
                        message += " Generated datasets are available for download below."
                        
                        # Extract just the filenames for our dataset links
                        for link in dataset_links:
                            link['filename'] = os.path.basename(link['path'])
                        
                        return render(request, 'processor/process_complete.html', {
                            'message': message,
                            'dataset_links': dataset_links,
                            'stats': stats,
                            'output': output.getvalue()
                        })

                except Exception as e:
                    message = f"An error occurred: {e}"
                    print(f"Processing error: {e}")
                    import traceback
                    traceback.print_exc()  # Print the full traceback for debugging

                if zip_file:
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        print(f"Temp directory '{temp_dir}' removed successfully.")
                    except OSError as e:
                         print(f"Error removing temp directory: {e}")
            else:
                message = 'Form is not valid. Please check the inputs.'
                print(f"Form errors: {form.errors}")
        else:
            # Create form with custom choices for initial page load
            form = ProcessingForm()
            form.fields['selected_participants'].choices = participant_choices
            form.fields['selected_words'].choices = word_choices

        return render(request, 'processor/process_data.html', {
            'form': form, 
            'message': message, 
            'output': output.getvalue(),
            'available_participants': available_participants,
            'available_words': available_words
        })

    finally: # Very important to restore sys.stdout
        sys.stdout = old_stdout # Restore the original stdout


def download_dataset(request, filename):
    """View to handle dataset downloads"""
    file_path = os.path.join(settings.BASE_DIR, 'Trials_data', filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    return HttpResponse("File not found", status=404)