def calculate_gc_content(sequence):
    
    gc_count = sequence.count('G') + sequence.count('C')
    total_length = len(sequence)
    gc_content = (gc_count / total_length) * 100 if total_length > 0 else 0
    return gc_content
