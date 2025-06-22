# trials/test_import_fix.py
# Test script to verify the EEG import fix

import os
import sys
import django

if __name__ == "__main__":
    # Setup Django environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
    django.setup()
    
    print("=== Testing Import Fix ===")
    
    # Test 1: Import through module (correct way)
    from trials import data_collection
    headset_via_module = data_collection.cyHeadset
    print(f"Via module: cyHeadset is None = {headset_via_module is None}")
    
    # Test 2: Direct import (problematic way)
    from trials.data_collection import cyHeadset as headset_direct
    print(f"Direct import: cyHeadset is None = {headset_direct is None}")
    
    # Test 3: Test visual_data_collection new function
    from trials.visual_data_collection import get_current_headset
    headset_via_function = get_current_headset()
    print(f"Via function: cyHeadset is None = {headset_via_function is None}")
    
    # Test 4: Compare identity
    print(f"Module vs Direct: Same object = {headset_via_module is headset_direct}")
    print(f"Module vs Function: Same object = {headset_via_module is headset_via_function}")
    
    # Test 5: If any are not None, show details
    for name, headset in [("Module", headset_via_module), ("Direct", headset_direct), ("Function", headset_via_function)]:
        if headset is not None:
            print(f"{name} headset details:")
            print(f"  Type: {type(headset)}")
            print(f"  Has hid: {hasattr(headset, 'hid')}")
            if hasattr(headset, 'hid'):
                print(f"  HID is None: {headset.hid is None}")
    
    print("=== Test Complete ===")