#!/usr/bin/env python3
"""
Test script demonstrating the model-based reinforcement learning approach 
for piano fingering on Fur Elise.
"""

import os
import sys
import time
from music21 import converter, note, chord

# Import the predict_fingering function
from predict_fingering import predict_fingering_reinforcement_learning

# Path to the test music file
TEST_FILE = "Fr_Elise.mxl"
OUTPUT_FILE = "Fr_Elise_rl_fingering.musicxml"

def main():
    print("=== Piano Fingering Generation with Model-Based Reinforcement Learning ===")
    print(f"Processing: {TEST_FILE}")
    
    start_time = time.time()
    
    # Process the file with reinforcement learning
    score, _, _ = predict_fingering_reinforcement_learning(TEST_FILE, OUTPUT_FILE)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Results saved to: {OUTPUT_FILE}")
    
    # Display a summary of the fingerings
    analyze_fingerings(OUTPUT_FILE)

def analyze_fingerings(file_path):
    """
    Analyze the fingering annotations in the output file
    """
    print("\n=== Fingering Analysis ===")
    
    # Parse the score
    score = converter.parse(file_path)
    
    # Count fingerings by hand
    right_hand_fingerings = {}
    left_hand_fingerings = {}
    
    # Process right hand (usually first part)
    if len(score.parts) > 0:
        right_part = score.parts[0]
        for element in right_part.flatten().notesAndRests:
            if isinstance(element, note.Note):
                for articulation in element.articulations:
                    if hasattr(articulation, 'fingerNumber'):
                        finger = articulation.fingerNumber
                        right_hand_fingerings[finger] = right_hand_fingerings.get(finger, 0) + 1
            elif isinstance(element, chord.Chord):
                for articulation in element.articulations:
                    if hasattr(articulation, 'fingerNumber'):
                        finger = articulation.fingerNumber
                        right_hand_fingerings[finger] = right_hand_fingerings.get(finger, 0) + 1
    
    # Process left hand (usually second part)
    if len(score.parts) > 1:
        left_part = score.parts[1]
        for element in left_part.flatten().notesAndRests:
            if isinstance(element, note.Note):
                for articulation in element.articulations:
                    if hasattr(articulation, 'fingerNumber'):
                        finger = articulation.fingerNumber
                        left_hand_fingerings[finger] = left_hand_fingerings.get(finger, 0) + 1
            elif isinstance(element, chord.Chord):
                for articulation in element.articulations:
                    if hasattr(articulation, 'fingerNumber'):
                        finger = articulation.fingerNumber
                        left_hand_fingerings[finger] = left_hand_fingerings.get(finger, 0) + 1
    
    # Display the results
    print("\nRight Hand Fingering Distribution:")
    total_right = sum(right_hand_fingerings.values())
    if total_right > 0:
        for finger in sorted(right_hand_fingerings.keys()):
            count = right_hand_fingerings[finger]
            print(f"  Finger {finger}: {count} times ({count/total_right*100:.1f}%)")
    else:
        print("  No fingerings found")
    
    print("\nLeft Hand Fingering Distribution:")
    total_left = sum(left_hand_fingerings.values())
    if total_left > 0:
        for finger in sorted(left_hand_fingerings.keys()):
            count = left_hand_fingerings[finger]
            print(f"  Finger {finger}: {count} times ({count/total_left*100:.1f}%)")
    else:
        print("  No fingerings found")
    
    print(f"\nTotal fingerings: {total_right + total_left}")

if __name__ == "__main__":
    main() 