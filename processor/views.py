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
import datetime


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

                # Create a timestamped output directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir_name = f'processed_{timestamp}'
                output_dir = os.path.join(root_dir, output_dir_name)
                os.makedirs(output_dir, exist_ok=True)

                try:
                    stats = process_trial_data(
                        root_dir=root_dir,
                        output_dir=output_dir,  # Pass the new output directory
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
                        
                        return render(request, 'processor/process_complete.html', {
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

        return render(request, 'processor/process_data.html', {
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