# trials/debug_eeg.py
# Simple debug script to test EEG connection for visual trials

import os
import sys
import django

# Setup Django environment
if __name__ == "__main__":
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Setup Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
    django.setup()
    
    # Now import Django modules
    from trials.data_collection import cyHeadset, SENSOR_ORDER, initialize_eeg
    
    print("=== EEG Debug Information ===")
    print(f"cyHeadset is None: {cyHeadset is None}")
    
    if cyHeadset is not None:
        print(f"cyHeadset type: {type(cyHeadset)}")
        print(f"cyHeadset has hid attribute: {hasattr(cyHeadset, 'hid')}")
        
        if hasattr(cyHeadset, 'hid'):
            print(f"cyHeadset.hid is None: {cyHeadset.hid is None}")
            if cyHeadset.hid is not None:
                print(f"cyHeadset.hid type: {type(cyHeadset.hid)}")
    
    print(f"SENSOR_ORDER length: {len(SENSOR_ORDER)}")
    print(f"SENSOR_ORDER: {SENSOR_ORDER}")
    
    # Try to initialize if not already done
    if cyHeadset is None or (hasattr(cyHeadset, 'hid') and cyHeadset.hid is None):
        print("\nTrying to initialize EEG headset...")
        success = initialize_eeg()
        print(f"Initialization successful: {success}")
        
        if success and cyHeadset is not None:
            print(f"After init - cyHeadset type: {type(cyHeadset)}")
            print(f"After init - cyHeadset.hid is None: {cyHeadset.hid is None if hasattr(cyHeadset, 'hid') else 'No hid attribute'}")
    
    # Test data collection
    if cyHeadset is not None and hasattr(cyHeadset, 'hid') and cyHeadset.hid is not None:
        print("\nTesting data collection...")
        try:
            # Clear buffer
            cyHeadset.clear_data()
            print("Buffer cleared successfully")
            
            # Try to get 5 data points
            for i in range(5):
                data = cyHeadset.get_data()
                if data is not None:
                    values = data.split(',')
                    print(f"Sample {i+1}: {len(values)} values - {data[:50]}...")
                else:
                    print(f"Sample {i+1}: No data received")
                    
        except Exception as e:
            print(f"Error during data collection test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nCannot test data collection - EEG headset not properly initialized")
    
    print("\n=== Debug Complete ===")