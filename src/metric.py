def calculate_recognition_rate(predictions, ground_truths):
    """
    Calculate Recognition Rate metric for license plate recognition.
    
    Recognition Rate = Number of Correct Tracks / Total Tracks in the Test Set
    
    A track is considered correct only when all predicted characters match 
    the ground truth exactly. If any character differs, the track is not counted as correct.
    
    Args:
        predictions (dict): Dictionary where keys are track IDs and values are predicted plate texts
        ground_truths (dict): Dictionary where keys are track IDs and values are ground truth plate texts
        
    Returns:
        float: Recognition rate (between 0.0 and 1.0)
        int: Number of correct tracks
        int: Total number of tracks
    """
    total_tracks = len(ground_truths)
    correct_tracks = 0
    
    # Check that predictions and ground truths have the same tracks
    missing_tracks = set(ground_truths.keys()) - set(predictions.keys())
    extra_tracks = set(predictions.keys()) - set(ground_truths.keys())
    
    if missing_tracks:
        print(f"Warning: Predictions missing for tracks: {missing_tracks}")
    if extra_tracks:
        print(f"Warning: Extra predictions for tracks not in ground truth: {extra_tracks}")
    
    # Calculate correct tracks
    for track_id, gt_text in ground_truths.items():
        if track_id in predictions:
            pred_text = predictions[track_id]
            if pred_text == gt_text:
                correct_tracks += 1
    
    recognition_rate = correct_tracks / total_tracks if total_tracks > 0 else 0.0
    
    return recognition_rate, correct_tracks, total_tracks


def calculate_character_recognition_rate(predictions, ground_truths):
    """
    Calculate Character Recognition Rate metric for license plate recognition.
    
    Character Recognition Rate = Total Correct Characters / Total Characters in All Tracks
    
    Measures the percentage of individual characters that are correctly recognized
    across all tracks.
    
    Args:
        predictions (dict): Dictionary where keys are track IDs and values are predicted plate texts
        ground_truths (dict): Dictionary where keys are track IDs and values are ground truth plate texts
        
    Returns:
        float: Character recognition rate (between 0.0 and 1.0)
        int: Number of correct characters
        int: Total number of characters
    """
    total_characters = 0
    correct_characters = 0
    
    # Check that predictions and ground truths have the same tracks
    missing_tracks = set(ground_truths.keys()) - set(predictions.keys())
    extra_tracks = set(predictions.keys()) - set(ground_truths.keys())
    
    if missing_tracks:
        print(f"Warning: Predictions missing for tracks: {missing_tracks}")
    if extra_tracks:
        print(f"Warning: Extra predictions for tracks not in ground truth: {extra_tracks}")
    
    # Calculate correct characters
    for track_id, gt_text in ground_truths.items():
        if track_id in predictions:
            pred_text = predictions[track_id]
            
            # Compare characters at each position (consider minimum length to avoid errors)
            min_length = min(len(gt_text), len(pred_text))
            for gt_char, pred_char in zip(gt_text[:min_length], pred_text[:min_length]):
                if gt_char == pred_char:
                    correct_characters += 1
            
            # Add total characters from ground truth
            total_characters += len(gt_text)
    
    character_rate = correct_characters / total_characters if total_characters > 0 else 0.0
    
    return character_rate, correct_characters, total_characters


def test_metrics():
    """Test function for all metrics."""
    # Sample test data
    ground_truths = {
        "track_10043": "ABC123",
        "track_10095": "XYZ789",
        "track_10112": "DEF456"
    }
    
    print("=== Test 1: All predictions correct ===")
    predictions1 = {
        "track_10043": "ABC123",
        "track_10095": "XYZ789",
        "track_10112": "DEF456"
    }
    rate1, correct1, total1 = calculate_recognition_rate(predictions1, ground_truths)
    char_rate1, char_correct1, char_total1 = calculate_character_recognition_rate(predictions1, ground_truths)
    print(f"Recognition Rate: {rate1:.2f}")
    print(f"Correct Tracks: {correct1}/{total1}")
    print(f"Character Recognition Rate: {char_rate1:.2f}")
    print(f"Correct Characters: {char_correct1}/{char_total1}")
    print()
    
    print("=== Test 2: One prediction incorrect ===")
    predictions2 = {
        "track_10043": "ABC123",
        "track_10095": "XYZ780",  # Incorrect character
        "track_10112": "DEF456"
    }
    rate2, correct2, total2 = calculate_recognition_rate(predictions2, ground_truths)
    char_rate2, char_correct2, char_total2 = calculate_character_recognition_rate(predictions2, ground_truths)
    print(f"Recognition Rate: {rate2:.2f}")
    print(f"Correct Tracks: {correct2}/{total2}")
    print(f"Character Recognition Rate: {char_rate2:.2f}")
    print(f"Correct Characters: {char_correct2}/{char_total2}")
    print()
    
    print("=== Test 3: All predictions incorrect ===")
    predictions3 = {
        "track_10043": "ABC124",
        "track_10095": "XYZ780",
        "track_10112": "DEF457"
    }
    rate3, correct3, total3 = calculate_recognition_rate(predictions3, ground_truths)
    char_rate3, char_correct3, char_total3 = calculate_character_recognition_rate(predictions3, ground_truths)
    print(f"Recognition Rate: {rate3:.2f}")
    print(f"Correct Tracks: {correct3}/{total3}")
    print(f"Character Recognition Rate: {char_rate3:.2f}")
    print(f"Correct Characters: {char_correct3}/{char_total3}")
    print()
    
    print("=== Test 4: Missing track in predictions ===")
    predictions4 = {
        "track_10043": "ABC123",
        "track_10095": "XYZ789"
    }
    rate4, correct4, total4 = calculate_recognition_rate(predictions4, ground_truths)
    char_rate4, char_correct4, char_total4 = calculate_character_recognition_rate(predictions4, ground_truths)
    print(f"Recognition Rate: {rate4:.2f}")
    print(f"Correct Tracks: {correct4}/{total4}")
    print(f"Character Recognition Rate: {char_rate4:.2f}")
    print(f"Correct Characters: {char_correct4}/{char_total4}")
    print()


if __name__ == "__main__":
    test_metrics()
