# trials/test_visual_eeg_flow.py
# Test script to simulate the visual EEG collection flow

import os
import sys
import django

if __name__ == "__main__":
    # Setup Django environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NeuroPi.settings')
    django.setup()
    
    print("=== Testing Visual EEG Flow ===")
    
    # Step 1: Check initial state
    from trials import data_collection
    from trials.visual_data_collection import get_current_headset, ensure_eeg_initialized
    
    print(f"1. Initial state - cyHeadset is None: {data_collection.cyHeadset is None}")
    
    # Step 2: Test ensure_eeg_initialized function
    print("\n2. Testing ensure_eeg_initialized()...")
    try:
        headset = ensure_eeg_initialized()
        print(f"   Success! Headset type: {type(headset)}")
        print(f"   Has hid: {hasattr(headset, 'hid')}")
        if hasattr(headset, 'hid'):
            print(f"   HID is None: {headset.hid is None}")
            
        # Test data collection
        if headset and hasattr(headset, 'hid') and headset.hid is not None:
            print("   Testing data collection...")
            headset.clear_data()
            test_data = headset.get_data()
            print(f"   Got test data: {test_data is not None}")
            if test_data:
                values = test_data.split(',')
                print(f"   Data length: {len(values)} values")
                
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test visual trial collection setup
    print("\n3. Testing visual trial collection setup...")
    try:
        from trials.models import WordSet, WordSetItem, VisualTrialSession
        
        # Create a test word set if needed
        word_set, created = WordSet.objects.get_or_create(
            name="Test Set",
            defaults={
                'description': 'Test word set for debugging',
                'is_active': True
            }
        )
        
        if created or word_set.words.count() == 0:
            # Add some test words
            test_words = ['test', 'debug', 'run']
            for i, word in enumerate(test_words):
                WordSetItem.objects.get_or_create(
                    word_set=word_set,
                    word=word,
                    defaults={'order': i}
                )
            print(f"   Created test word set with {len(test_words)} words")
        else:
            print(f"   Using existing word set with {word_set.words.count()} words")
        
        # Create a test session
        session = VisualTrialSession.objects.create(
            participant_name="TestParticipant",
            word_set=word_set,
            word_display_duration=3000,
            rest_duration=2000,
            repetitions_per_word=5
        )
        print(f"   Created test session: {session.id}")
        
        # Test the visual collection start
        from trials.visual_data_collection import start_visual_trial_collection
        print("   Attempting to start visual trial collection...")
        
        collector = start_visual_trial_collection(session.id)
        print(f"   Success! Collector created: {type(collector)}")
        print(f"   Collection active: {collector.is_collecting}")
        
        # Let it run for a few seconds
        import time
        print("   Collecting data for 3 seconds...")
        time.sleep(3)
        
        print(f"   Samples collected: {len(collector.eeg_data)}")
        print(f"   Current word: {collector.current_word}")
        
        # Test word changing
        collector.set_current_word("test")
        time.sleep(1)
        collector.set_rest_period()
        time.sleep(1)
        
        print(f"   Final samples collected: {len(collector.eeg_data)}")
        
        # Stop collection
        from trials.visual_data_collection import stop_visual_trial_collection
        success, result = stop_visual_trial_collection()
        print(f"   Stopped collection: {success}")
        print(f"   Final result: {result}")
        
        # Clean up test session
        session.delete()
        if created:
            word_set.delete()
        print("   Cleaned up test data")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")