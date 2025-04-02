# NeuroPi EEG Data Filename Dictionary

This document explains how processing parameters are encoded in the filenames of cleaned EEG data files.

## File Naming Convention

Processed files follow this pattern:
```
[base_name]_[filter_codes]_[transformer_codes]_[split_codes]_[timestamp].csv
```

Example:
```
cleaned_data_BP4.0-50.0o5_N50.0_ICA-inf_Norm_TF_S40_M20_Str_Split_T20_Str_S42_20250402_123456.csv
```

## Filter Codes

### Frequency Filters
- `BP{low}-{high}o{order}` - Bandpass filter
  - Example: `BP4.0-50.0o5` = Bandpass filter 4-50Hz, order 5
  
- `HP{cutoff}o{order}` - Highpass filter 
  - Example: `HP4.0o5` = Highpass filter 4Hz, order 5
  
- `LP{cutoff}o{order}` - Lowpass filter
  - Example: `LP50.0o5` = Lowpass filter 50Hz, order 5

### Notch Filter
- `N{freq}` - Notch filter to remove specific frequency
  - Example: `N50.0` = Notch filter at 50Hz

### Artifact Removal
- `ICA-{method}` - Independent Component Analysis
  - Methods: `fas` (FastICA), `inf` (Infomax), `pic` (Picard)
  - Example: `ICA-inf` = ICA using Infomax method

### Other Transformations
- `Norm` - Normalization (Z-score) applied
- `Out{threshold}` - Outlier removal 
  - Example: `Out3.0` = Outliers removed at 3.0 standard deviations

### No Filters
- `RAW` - No filters applied

## Transformer Codes

Codes related to CNN-Transformer preparation (only present if transformer preparation was enabled):

- `TF` - Indicates data processed for CNN-Transformer
- `S{number}` - Sequence length
  - Example: `S40` = Sequence length of 40 samples
- `M{number}` - Minimum segment length
  - Example: `M20` = Minimum segment length of 20 samples
- `Str` - Structured format used
- `Bal` - Class balancing applied

## Split Codes

Codes related to train-test splitting (only present if splitting was enabled):

- `Split` - Indicates train-test split was created
- `T{number}` - Test size percentage
  - Example: `T20` = 20% test size
- `Str` - Stratified by word/event
- `Rnd` - Random split (non-stratified)
- `S{number}` - Random seed used
  - Example: `S42` = Random seed 42

## Timestamp

The timestamp is in format: `YYYYMMDD_HHMMSS`
Example: `20250402_123456` = April 2, 2025 at 12:34:56

## Complete Examples

1. `cleaned_data_BP4.0-50.0o5_N50.0_ICA-inf_Norm_20250402_123456.csv`
   - Bandpass filter 4-50Hz, order 5
   - Notch filter at 50Hz
   - ICA with Infomax method
   - Z-score normalization
   - Processed on April 2, 2025 at 12:34:56

2. `cleaned_data_BP4.0-50.0o5_N50.0_TF_S40_M20_Str_Split_T20_Str_S42_20250402_123456.csv`
   - Bandpass filter 4-50Hz, order 5
   - Notch filter at 50Hz
   - CNN-Transformer preparation with sequence length 40, min segment 20
   - Train-test split with 20% test size, stratified by word, seed 42
   - Processed on April 2, 2025 at 12:34:56

3. `cleaned_data_HP1.0o3_LP30.0o3_Out2.5_20250402_123456.csv`
   - Highpass filter at 1Hz, order 3
   - Lowpass filter at 30Hz, order 3
   - Outlier removal at 2.5 standard deviations
   - Processed on April 2, 2025 at 12:34:56

## Using in Models

For model training and prediction, load the matching configuration using:

```python
from eeg_utils import parse_processed_filename, load_processing_config

# From filename
config = parse_processed_filename('cleaned_data_BP4.0-50.0o5_N50.0_ICA-inf_Norm_20250402_123456.csv')

# Or from JSON config file
config = load_processing_config('processing_config.json')

# Apply the same processing during prediction
filtered_data = apply_filters(new_data, config['filter_config'])
```

This ensures consistent processing between training and prediction.