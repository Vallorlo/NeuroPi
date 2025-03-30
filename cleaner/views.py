from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.contrib import messages
from django.utils.html import mark_safe

from .forms import CleaningForm
from .clean_eeg import clean_eeg_data

import os
import io
import sys
import json
import traceback
import zipfile
from datetime import datetime
import csv
import pandas as pd

def clean_data_view(request):
    """Main view for the EEG cleaning tool."""
    print("clean_data_view called")  # Debug: Start of view
    
    # Redirect stdout to capture debugging output
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    form = CleaningForm()
    context = {
        'form': form,
        'message': '',
        'output': '',
        'plots': None,
        'features': False,
        'error': None,  # Add a dedicated error field to the context
        'debug_info': None  # Add a field for debugging information
    }
    
    try:
        if request.method == 'POST':
            print("POST request received")  # Debug: POST request
            
            # Check if this is an AJAX request to get columns from a selected file
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' and 'get_columns' in request.POST:
                input_file = request.POST.get('input_file')
                if input_file and os.path.exists(input_file):
                    try:
                        # Read the first few rows to get columns
                        df = pd.read_csv(input_file, nrows=5)
                        columns = df.columns.tolist()
                        
                        # Categorize columns
                        sensor_columns = [col for col in columns if col not in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                                        and not col.endswith('_event')]
                        event_columns = [col for col in columns if col.endswith('_event')]
                        metadata_columns = [col for col in ['Timestamp', 'COUNTER', 'participant_id', 'word', 'stage', 'attempt', 'word_label'] 
                                          if col in columns]
                        
                        # Return the categorized columns
                        return JsonResponse({
                            'success': True,
                            'columns': columns,
                            'sensor_columns': sensor_columns,
                            'event_columns': event_columns,
                            'metadata_columns': metadata_columns
                        })
                    except Exception as e:
                        print(f"Error getting columns: {str(e)}")
                        traceback.print_exc()
                        return JsonResponse({
                            'success': False,
                            'error': str(e)
                        })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'Invalid file path'
                    })
            
            form = CleaningForm(request.POST)
            
            if form.is_valid():
                print("Form is valid")  # Debug: Form validation
                
                # Extract form data
                input_file = form.cleaned_data['input_file']
                output_file_name = form.cleaned_data['output_file_name']
                
                # Get columns_to_drop from form - UPDATED to handle the new way
                columns_to_drop = form.cleaned_data.get('columns_to_drop', [])
                # Make sure it's a list
                if not isinstance(columns_to_drop, list):
                    if isinstance(columns_to_drop, str):
                        columns_to_drop = [col.strip() for col in columns_to_drop.split(',') if col.strip()]
                    else:
                        columns_to_drop = []
                
                print(f"Columns to drop: {columns_to_drop}")
                
                # Filter settings
                apply_bandpass = form.cleaned_data['apply_bandpass']
                lowcut = form.cleaned_data['lowcut']
                highcut = form.cleaned_data['highcut']
                bandpass_order = form.cleaned_data['bandpass_order']
                
                apply_highpass = form.cleaned_data['apply_highpass']
                highpass_cutoff = form.cleaned_data['highpass_cutoff']
                highpass_order = form.cleaned_data['highpass_order']
                
                apply_lowpass = form.cleaned_data['apply_lowpass']
                lowpass_cutoff = form.cleaned_data['lowpass_cutoff']
                lowpass_order = form.cleaned_data['lowpass_order']
                
                apply_notch = form.cleaned_data['apply_notch']
                notch_freq = form.cleaned_data['notch_freq']
                notch_quality = form.cleaned_data['notch_quality']
                
                apply_ica = form.cleaned_data['apply_ica']
                ica_method = form.cleaned_data['ica_method']
                random_seed = form.cleaned_data['random_seed']
                
                generate_plots = form.cleaned_data.get('generate_plots', False)
                extract_features = form.cleaned_data.get('extract_features', False)
                regions_of_interest = form.cleaned_data.get('regions_of_interest', ['all'])
                

                create_train_test_split = form.cleaned_data.get('create_train_test_split', False)
                test_size = form.cleaned_data.get('test_size', 0.2)
                random_state = form.cleaned_data.get('random_state', 42)
                stratify_by_word = form.cleaned_data.get('stratify_by_word', True)
                
                prepare_for_transformer = form.cleaned_data.get('prepare_for_transformer', False)
                use_structured_format = form.cleaned_data.get('use_structured_format', True)
                sequence_length = form.cleaned_data.get('sequence_length', 40)
                min_segment_length = form.cleaned_data.get('min_segment_length', 20)
                balance_classes = form.cleaned_data.get('balance_classes', False)

                include_channels = form.cleaned_data.get('include_channels', None)
                compute_band_powers = form.cleaned_data.get('compute_band_powers', False)
                normalize_data = form.cleaned_data.get('normalize_data', False)
                remove_outliers = form.cleaned_data.get('remove_outliers', False)
                outlier_threshold = form.cleaned_data.get('outlier_threshold', 3.0)
                
                # Automatically detect & fix signal issues
                check_signal_quality = True
                auto_scale = False  # Changed to False to preserve input values

                # Create output directory with timestamp to prevent overwrites
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(settings.BASE_DIR, 'Trials_data', f'cleaned_{timestamp}')
                os.makedirs(output_dir, exist_ok=True)
                
                output_file_path = os.path.join(output_dir, output_file_name)
                print(f"Output file path: {output_file_path}")  # Debug: Check output path

                try:
                    # Print debugging information about what's being dropped
                    print(f"Columns to drop: {columns_to_drop}")
                    if columns_to_drop:
                        event_columns_to_drop = [col for col in columns_to_drop if col.endswith('_event')]
                        other_columns_to_drop = [col for col in columns_to_drop if not col.endswith('_event')]
                        print(f"Event columns to drop: {event_columns_to_drop}")
                        print(f"Other columns to drop: {other_columns_to_drop}")
                    
                    # Call the EEG cleaning function with all parameters
                    result = clean_eeg_data(
                        input_file=input_file,
                        output_file=output_file_path,
                        apply_bandpass=apply_bandpass,
                        lowcut=lowcut,
                        highcut=highcut,
                        bandpass_order=bandpass_order,
                        apply_highpass=apply_highpass,
                        highpass_cutoff=highpass_cutoff,
                        highpass_order=highpass_order,
                        apply_lowpass=apply_lowpass,
                        lowpass_cutoff=lowpass_cutoff,
                        lowpass_order=lowpass_order,
                        apply_notch=apply_notch,
                        notch_freq=notch_freq,
                        notch_quality=notch_quality,
                        apply_ica=apply_ica,
                        ica_method=ica_method,
                        random_seed=random_seed,
                        generate_plots=generate_plots,
                        extract_features=extract_features,
                        check_signal_quality=check_signal_quality,
                        auto_scale=auto_scale,
                        create_train_test_split=create_train_test_split,
                        test_size=test_size,
                        random_state=random_state,
                        stratify_by_word=stratify_by_word,
                        include_channels=include_channels,
                        compute_band_powers=compute_band_powers,
                        normalize_data=normalize_data,
                        remove_outliers=remove_outliers,
                        outlier_threshold=outlier_threshold,
                        prepare_for_transformer=prepare_for_transformer,
                        sequence_length=sequence_length,
                        min_segment_length=min_segment_length,
                        use_structured_format=use_structured_format,
                        balance_classes=balance_classes,
                        columns_to_drop=columns_to_drop
                    )
                    
                    print(f"clean_eeg_data returned: {result}")  # Debug: Check return value
                    
                    if result.get('status') == 'error':
                        # Handle error result from clean_eeg_data
                        error_message = result.get('message', 'An unknown error occurred during processing')
                        print(f"Error returned from clean_eeg_data: {error_message}")
                        
                        # Instead of refreshing, show the form with the error message
                        context = {
                            'form': form,
                            'message': '',
                            'output': output.getvalue(),
                            'error': error_message,
                            'debug_info': f"Processing failed. See error message and logs for details."
                        }
                        return render(request, 'cleaner/clean_data.html', context)
                    
                    if isinstance(result, dict) and result.get('status') == 'success':
                        message = result.get('message', 'Data cleaned successfully!')
                        plots = result.get('plots')
                        features = result.get('features', False)
                        
                        # Create a ZIP file with all outputs for the user to download
                        zip_filename = os.path.join(output_dir, 'cleaned_data_package.zip')
                        with zipfile.ZipFile(zip_filename, 'w') as zipf:
                            # Add the cleaned CSV file
                            zipf.write(output_file_path, os.path.basename(output_file_path))
                            
                            # Add diagnostic plots if generated
                            if plots:
                                for plot_name, plot_path in plots.items():
                                    if os.path.exists(plot_path):
                                        zipf.write(plot_path, os.path.basename(plot_path))
                            
                            # Add features file if extracted
                            feature_file = os.path.join(os.path.dirname(output_file_path), 'speech_features.npz')
                            if features and os.path.exists(feature_file):
                                zipf.write(feature_file, os.path.basename(feature_file))
                                
                            # Add signal quality report
                            if result.get('signal_stats'):
                                report_path = os.path.join(output_dir, 'signal_quality_report.txt')
                                with open(report_path, 'w') as f:
                                    f.write("EEG Signal Quality Report\n")
                                    f.write("=======================\n\n")
                                    f.write("**This report is based on the INPUT data before any filtering**\n\n")
                                    
                                    stats = result['signal_stats']
                                    f.write(f"Min value: {stats.get('min', 'N/A')}\n")
                                    f.write(f"Max value: {stats.get('max', 'N/A')}\n")
                                    f.write(f"Mean value: {stats.get('mean', 'N/A')}\n")
                                    f.write(f"Standard deviation: {stats.get('std', 'N/A')}\n")
                                    f.write(f"Zero values: {stats.get('zeros_percent', 'N/A')}%\n\n")
                                    
                                    if stats.get('warnings'):
                                        f.write("Warnings:\n")
                                        for warning in stats['warnings']:
                                            f.write(f"- {warning}\n")
                                            
                                    if 'min_scaled' in stats:
                                        f.write("\nAfter auto-scaling:\n")
                                        f.write(f"Min value: {stats.get('min_scaled', 'N/A')}\n")
                                        f.write(f"Max value: {stats.get('max_scaled', 'N/A')}\n")
                                        f.write(f"Mean value: {stats.get('mean_scaled', 'N/A')}\n")
                                        f.write(f"Standard deviation: {stats.get('std_scaled', 'N/A')}\n")
                                
                                zipf.write(report_path, os.path.basename(report_path))
                        
                        # Add train-test split files to zip if they exist
                        if result.get('train_file') and os.path.exists(result.get('train_file')):
                            with zipfile.ZipFile(zip_filename, 'a') as zipf:
                                zipf.write(result['train_file'], os.path.basename(result['train_file']))
                        
                        if result.get('test_file') and os.path.exists(result.get('test_file')):
                            with zipfile.ZipFile(zip_filename, 'a') as zipf:
                                zipf.write(result['test_file'], os.path.basename(result['test_file']))
                        
                        if result.get('transformer_dataset'):
                            transformer_file = result.get('transformer_file')
                            if transformer_file and os.path.exists(transformer_file):
                                with zipfile.ZipFile(zip_filename, 'a') as zipf:
                                    zipf.write(transformer_file, os.path.basename(transformer_file))
                                    
                                # Add to download links
                                result['transformer_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(transformer_file)}"
                            
                            # Add train/test files if they exist
                            if result.get('transformer_train_file') and os.path.exists(result.get('transformer_train_file')):
                                with zipfile.ZipFile(zip_filename, 'a') as zipf:
                                    zipf.write(result['transformer_train_file'], os.path.basename(result['transformer_train_file']))
                                    
                                result['transformer_train_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(result['transformer_train_file'])}"
                                
                            if result.get('transformer_test_file') and os.path.exists(result.get('transformer_test_file')):
                                with zipfile.ZipFile(zip_filename, 'a') as zipf:
                                    zipf.write(result['transformer_test_file'], os.path.basename(result['transformer_test_file']))
                                    
                                result['transformer_test_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(result['transformer_test_file'])}"

                        # Prepare data for the clean_complete.html template
                        # Read a sample of the cleaned data for preview
                        data_preview = []
                        try:
                            # Read the first 10 rows of the cleaned file
                            with open(output_file_path, 'r') as f:
                                reader = csv.reader(f)
                                headers = next(reader)  # Get header row
                                
                                # Determine timestamp column and event columns
                                timestamp_col = 0  # Default to first column
                                if 'Timestamp' in headers:
                                    timestamp_col = headers.index('Timestamp')
                                
                                event_cols = [i for i, h in enumerate(headers) if h.endswith('_event')]
                                
                                # Check if we have word_label column
                                word_label_col = -1
                                if 'word_label' in headers:
                                    word_label_col = headers.index('word_label')
                                
                                # Get sensor column indices
                                sensor_cols = []
                                for channel in result.get('channels_processed', []):
                                    if channel in headers:
                                        sensor_cols.append(headers.index(channel))
                                
                                # If no sensor columns found but we have time_ch columns, use those
                                if not sensor_cols and any('time' in h and 'ch' in h for h in headers):
                                    sensor_cols = [i for i, h in enumerate(headers) if 'time' in h and 'ch' in h][:8]  # First 8 channels
                                
                                # Read up to 10 rows
                                for i, row in enumerate(reader):
                                    if i >= 10:  # Only get first 10 rows
                                        break
                                        
                                    # Check if we have an event in any event column
                                    has_event = False
                                    
                                    if event_cols:
                                        has_event = any(row[col].lower() == 'true' for col in event_cols if col < len(row))
                                    elif word_label_col >= 0:
                                        # If we have word_label column, check if it's not 'sil'
                                        if word_label_col < len(row) and row[word_label_col] != 'sil':
                                            has_event = True
                                    
                                    # Format the preview data
                                    preview_row = {
                                        'timestamp': float(row[timestamp_col]) if timestamp_col < len(row) and row[timestamp_col] and row[timestamp_col] != 'nan' else 0,
                                        'values': [],
                                        'event': has_event
                                    }
                                    
                                    # Add sensor values
                                    for col in sensor_cols:
                                        if col < len(row) and row[col] and row[col] != 'nan':
                                            preview_row['values'].append(int(float(row[col])))
                                        else:
                                            preview_row['values'].append(0)
                                    
                                    # If we have word_label column, add it
                                    if word_label_col >= 0 and word_label_col < len(row):
                                        preview_row['word_label'] = row[word_label_col]
                                    
                                    data_preview.append(preview_row)
                        except Exception as e:
                            print(f"Error creating data preview: {e}")
                            traceback.print_exc()
                            data_preview = []
                        
                        # Add file URLs for downloading
                        result['output_file_name'] = os.path.basename(output_file_path)
                        result['output_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(output_file_path)}"

                        # For any train/test files:
                        if result.get('train_file'):
                            result['train_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(result['train_file'])}"

                        if result.get('test_file'):
                            result['test_file_url'] = f"/cleaner/download/{os.path.basename(output_dir)}/{os.path.basename(result['test_file'])}"
                        
                        # Fix plots URLs if they exist
                        if plots:
                            # Convert full paths to relative URLs
                            for plot_key, plot_path in plots.items():
                                rel_path = os.path.relpath(plot_path, settings.BASE_DIR)
                                plots[plot_key] = f"/{rel_path.replace(os.sep, '/')}"
                        
                        # Render the completion template
                        return render(request, 'cleaner/clean_complete.html', {
                            'message': message,
                            'result': result,
                            'data_preview': data_preview,
                            'output': output.getvalue()
                        })
                        
                except Exception as e:
                    trace = traceback.format_exc()
                    error_message = f'An error occurred: {str(e)}'
                    print(f"Error during processing: {e}")
                    print(trace)
                    
                    # Instead of returning a JavaScript redirect, render the template with the error
                    context = {
                        'form': form, 
                        'message': error_message,
                        'output': output.getvalue(),
                        'error': str(e),
                        'debug_info': trace
                    }
                    return render(request, 'cleaner/clean_data.html', context)
            else:
                print("Form is NOT valid")  # Debug: Form invalid
                print(form.errors)  # Print Errors
                
                # Add form errors to the context
                context = {
                    'form': form,
                    'message': "Invalid form inputs. Please check the error messages.",
                    'output': output.getvalue(),
                    'error': "Form validation failed. Please correct the errors below.",
                    'form_errors': form.errors
                }
                return render(request, 'cleaner/clean_data.html', context)
                
        else:
            # Create form with custom choices for initial page load
            form = CleaningForm()

        # Check for existing processed data
        processed_dirs = [d for d in os.listdir(settings.TRIAL_DIR) 
                         if os.path.isdir(os.path.join(settings.TRIAL_DIR, d)) 
                         and d.startswith('processed_')]
        
        has_processed_data = len(processed_dirs) > 0
        latest_processed = max(processed_dirs, default=None) if processed_dirs else None

        # Update the context
        context.update({
            'form': form, 
            'message': context.get('message', ''), 
            'output': output.getvalue(),
            'has_processed_data': has_processed_data,
            'latest_processed': latest_processed
        })

        return render(request, 'cleaner/clean_data.html', context)

    except Exception as e:
        trace = traceback.format_exc()
        error_message = f"Unexpected error: {str(e)}"
        print(f"Exception in view: {e}")
        print(trace)
        
        context.update({
            'message': error_message,
            'output': output.getvalue() if 'output' in locals() else "",
            'error': str(e),
            'debug_info': trace
        })
    
    finally:
        # Restore stdout
        if 'old_stdout' in locals():
            sys.stdout = old_stdout
        
    return render(request, 'cleaner/clean_data.html', context)

def download_file_view(request, filepath):
    """View for downloading files."""
    try:
        # For security, validate the filepath is within the allowed directory
        base_dir = settings.BASE_DIR
        
        # Handle paths that might include subdirectories
        path_parts = filepath.split('/')
        
        if len(path_parts) > 1:
            # If the file is in a subdirectory
            subdir = path_parts[0]
            file_name = '/'.join(path_parts[1:])  # Join the rest back together
            full_path = os.path.join(base_dir, 'Trials_data', subdir, file_name)
        else:
            # Direct file in Trials_data directory
            full_path = os.path.join(base_dir, 'Trials_data', filepath)
        
        # If not found directly, search subdirectories
        if not os.path.exists(full_path):
            for root, dirs, files in os.walk(os.path.join(base_dir, 'Trials_data')):
                for file in files:
                    if file == path_parts[-1]:  # Match the filename (last part of the path)
                        candidate_path = os.path.join(root, file)
                        # Check if the path is valid
                        if os.path.abspath(candidate_path).startswith(os.path.abspath(base_dir)):
                            full_path = candidate_path
                            break
        
        # Check if the file exists and is within the base directory
        if not os.path.exists(full_path):
            return HttpResponse("File not found", status=404)
        
        # Validate path is within base directory for security
        requested_path = os.path.abspath(full_path)
        if not requested_path.startswith(base_dir):
            return HttpResponse("Access denied", status=403)
        
        with open(requested_path, 'rb') as f:
            response = HttpResponse(f.read())
            
            # Set the content type based on file extension
            file_ext = os.path.splitext(requested_path)[1].lower()
            if file_ext == '.csv':
                response['Content-Type'] = 'text/csv'
            elif file_ext == '.png':
                response['Content-Type'] = 'image/png'
            elif file_ext == '.jpg' or file_ext == '.jpeg':
                response['Content-Type'] = 'image/jpeg'
            elif file_ext == '.npz':
                response['Content-Type'] = 'application/octet-stream'
            elif file_ext == '.zip':
                response['Content-Type'] = 'application/zip'
            else:
                response['Content-Type'] = 'application/octet-stream'
            
            # Set filename for download
            filename = os.path.basename(requested_path)
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
        return response
    
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}", status=500)