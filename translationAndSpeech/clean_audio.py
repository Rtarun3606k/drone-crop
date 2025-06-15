#!/usr/bin/env python3
"""
Utility script to clean up the audio directory
"""

import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Clean up the audio directory')
    parser.add_argument('--keep-dir', action='store_true', help='Keep the directory structure but remove all files')
    args = parser.parse_args()
    
    audio_dir = os.path.join(os.getcwd(), 'audio')
    
    if not os.path.exists(audio_dir):
        print("Audio directory doesn't exist. Creating it...")
        os.makedirs(audio_dir)
        return
        
    if args.keep_dir:
        # Remove all files but keep the directory
        for file in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Removed {file_path}")
        print("All files removed, directory structure preserved.")
    else:
        # Remove the directory and recreate it
        shutil.rmtree(audio_dir)
        os.makedirs(audio_dir)
        print("Audio directory cleaned and recreated.")

if __name__ == "__main__":
    main()
