import torch
import sys

def load_model_safe(model_path):
    """Load a model safely in PyTorch 2.6+"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try loading with weights_only=False
        print(f"Loading model from {model_path} with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Model loaded successfully!")
        return checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Check PyTorch version
        pytorch_version = torch.__version__
        print(f"PyTorch version: {pytorch_version}")
        
        if int(pytorch_version.split('.')[0]) >= 2 and int(pytorch_version.split('.')[1]) >= 6:
            print("\nYou're using PyTorch 2.6+, which has a default of weights_only=True.")
            print("This can cause issues loading models saved with older PyTorch versions.")
            
            # Try with add_safe_globals if available
            try:
                print("\nTrying with safe_globals context manager...")
                from torch.serialization import safe_globals
                
                # These are common numpy functions used in saved models
                safe_numpy_funcs = [
                    'numpy.core.multiarray._reconstruct',
                    'numpy.core.multiarray.scalar',
                    'numpy.core._multiarray_umath',
                    'numpy.core.numeric'
                ]
                
                with safe_globals(safe_numpy_funcs):
                    checkpoint = torch.load(model_path, map_location=device)
                    print("Model loaded successfully with safe_globals!")
                    return checkpoint
            except Exception as e2:
                print(f"Error with safe_globals approach: {e2}")
                
                print("\nFallback: Try loading the model in a different Python script with an older PyTorch version.")
                return None
        else:
            print(f"Unexpected error for PyTorch {pytorch_version}. Check the model file integrity.")
            return None

def extract_model_info(checkpoint):
    """Extract and print basic info from a model checkpoint"""
    if checkpoint is None:
        return
    
    print("\nModel checkpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: <dictionary with {len(checkpoint[key])} items>")
        elif hasattr(checkpoint[key], 'shape'):
            print(f"  {key}: <tensor/array with shape {checkpoint[key].shape}>")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Print history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        print("\nTraining history:")
        for key in history:
            if isinstance(history[key], list):
                print(f"  {key}: <list with {len(history[key])} items>")
                if len(history[key]) > 0:
                    print(f"    Final value: {history[key][-1]}")
            else:
                print(f"  {key}: {history[key]}")

if __name__ == "__main__":
    # Check command line arguments for model path or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "speech_detection_model.pth"  # Default
    
    # Load the model
    checkpoint = load_model_safe(model_path)
    
    # Extract and print information
    if checkpoint:
        extract_model_info(checkpoint)