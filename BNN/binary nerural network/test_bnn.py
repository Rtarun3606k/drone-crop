#!/usr/bin/env python3
"""
Binary Neural Network - Plant Disease Prediction Testing Suite

This script provides an easy interface to run various prediction tests
on your trained Binary Neural Network model.

Usage Examples:
1. Quick test with 100 samples per class:
   python test_bnn.py quick

2. Standard test with 1000 samples per class:
   python test_bnn.py standard

3. Large test with 2000 samples per class:
   python test_bnn.py large

4. Custom test with specified sample size:
   python test_bnn.py custom 1500

5. Test ALL images in the dataset:
   python test_bnn.py complete

6. Interactive mode:
   python test_bnn.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predectorAllimges import (
    quick_test, 
    standard_test, 
    large_test, 
    complete_test, 
    custom_test,
    run_random_prediction_test,
    run_comprehensive_test
)

def print_usage():
    """Print usage instructions"""
    print(__doc__)

def main():
    """Main execution function"""
    
    if len(sys.argv) == 1:
        # Interactive mode
        print("🤖 Binary Neural Network - Plant Disease Prediction Testing")
        print("=" * 60)
        print("Available test modes:")
        print("1. Quick Test     - 100 samples per class")
        print("2. Standard Test  - 1000 samples per class")
        print("3. Large Test     - 2000 samples per class")
        print("4. Complete Test  - ALL images in dataset")
        print("5. Custom Test    - Specify sample size")
        print("=" * 60)
        
        choice = input("Select test mode (1-5): ").strip()
        
        if choice == "1":
            result = quick_test()
        elif choice == "2":
            result = standard_test()
        elif choice == "3":
            result = large_test()
        elif choice == "4":
            result = complete_test()
        elif choice == "5":
            try:
                sample_size = int(input("Enter sample size per class: "))
                result = custom_test(sample_size)
            except ValueError:
                print("❌ Invalid sample size. Using default 1000.")
                result = standard_test()
        else:
            print("❌ Invalid choice. Running standard test.")
            result = standard_test()
            
    elif len(sys.argv) == 2:
        # Command line mode
        mode = sys.argv[1].lower()
        
        if mode == "quick":
            result = quick_test()
        elif mode == "standard":
            result = standard_test()
        elif mode == "large":
            result = large_test()
        elif mode == "complete":
            result = complete_test()
        elif mode == "help" or mode == "-h" or mode == "--help":
            print_usage()
            return
        else:
            print(f"❌ Unknown mode: {mode}")
            print_usage()
            return
            
    elif len(sys.argv) == 3:
        # Custom mode with sample size
        mode = sys.argv[1].lower()
        
        if mode == "custom":
            try:
                sample_size = int(sys.argv[2])
                result = custom_test(sample_size)
            except ValueError:
                print(f"❌ Invalid sample size: {sys.argv[2]}")
                print("Using default standard test.")
                result = standard_test()
        else:
            print(f"❌ Unknown mode: {mode}")
            print_usage()
            return
    else:
        print("❌ Too many arguments.")
        print_usage()
        return
    
    # Print final result summary
    if result:
        print("\n✅ Test completed successfully!")
        print(f"Check the results directory for detailed reports and visualizations.")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main()
