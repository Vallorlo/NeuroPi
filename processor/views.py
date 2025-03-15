# processor/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from .forms import ProcessingForm
import os
import sys
import io
import zipfile
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

    try:
        if request.method == 'POST':
            form = ProcessingForm(request.POST, request.FILES)
            if form.is_valid():
                verbose = form.cleaned_data['verbose']
                create_visualizations = form.cleaned_data['create_visualizations']
                generate_dataset = form.cleaned_data['generate_dataset']
                silence_thresh = form.cleaned_data['silence_thresh']
                min_silence = form.cleaned_data['min_silence']
                timestamp_padding = form.cleaned_data['timestamp_padding']
                zip_file = request.FILES.get('zip_file')

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
                        selected_stages=selected_stages  # Pass selected stages
                    )
                    message = "Processing complete!"

                except Exception as e:
                    message = f"An error occurred: {e}"

                if zip_file:
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        print(f"Temp directory '{temp_dir}' removed successfully.")  #This will now go to output
                    except OSError as e:
                         print(f"Error removing temp directory: {e}") #This will now go to output
            else:
                message = 'Form is not valid. Please check the inputs.'
        else:
            form = ProcessingForm()

        return render(request, 'processor/process_data.html', {'form': form, 'message': message, 'output': output.getvalue()})

    finally: #Very important to restore sys.stdout
        sys.stdout = old_stdout # Restore the original stdout