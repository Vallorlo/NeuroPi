# Neuroπ - EEG Data Collection and Analysis Platform

Neuroπ is a comprehensive web application for collecting, processing, and analyzing EEG (Electroencephalography) data, with a specific focus on studying speech and motor patterns.

## Features

- **EEG Data Collection**: Interface with EEG devices to collect brain activity data
- **Trial Management**: Organize experiments into trials, words, and stages
- **Audio Recording**: Synchronize EEG data with audio recordings
- **Data Processing**: Advanced signal processing tools for EEG data
  - Bandpass, highpass, lowpass, and notch filtering
  - ICA (Independent Component Analysis) for artifact removal
  - Signal quality analysis
- **Data Cleaning**: Clean and prepare EEG data for analysis
  - Noise reduction
  - Outlier removal
  - Feature extraction
- **Visualization**: Interactive plots for EEG data analysis
- **Dataset Creation**: Generate combined datasets for machine learning applications
- **Train-Test Split**: Create ML-ready datasets with appropriate stratification

## Project Structure

- **NeuroPi**: Core application settings and configuration
- **trials**: Trial management and data collection
- **plot**: Data visualization tools
- **processor**: Data processing utilities
- **cleaner**: Signal cleaning and feature extraction

## Trial Stages

Neuroπ supports multiple stages of data collection for each word in a trial:

1. **Stage 1**: Vocal - Say the word out loud
2. **Stage 2**: Non-Vocal - Think about saying the word without speaking
3. **Stage 3**: Motor - Perform the action associated with the word without speaking
4. **Stage 4**: Motor + Vocal - Perform the action while saying the word
5. **Stage 5**: Motor + Non-Vocal - Perform the action while thinking about the word

## Installation

### Prerequisites

- Python 3.8+
- Django 5.1+
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/neuropi.git
   cd neuropi
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create database:
   ```
   python manage.py migrate
   ```

5. Create a superuser (optional):
   ```
   python manage.py createsuperuser
   ```

6. Run the development server:
   ```
   python manage.py runserver
   ```

7. Access the application at http://127.0.0.1:8000/

## Hardware Requirements

For EEG data collection, the application is designed to work with the Emotiv EPOC+ headset. The application interfaces with the device using the included data collection utilities.

## Data Flow

1. Capture raw EEG data through trials
2. Process the data to identify speech/motor events
3. Clean the signals to remove artifacts and noise
4. Create datasets for analysis or machine learning
5. Visualize results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Django community for the excellent web framework
- Emotiv for the EEG hardware compatibility
- All contributors and researchers in the field of EEG analysis

## Contact

For any questions or support, please open an issue on GitHub or contact the project maintainer.