#!/usr/bin/env python3
"""
Prepare submission file by converting normalized predictions to the correct format.
"""

import json
import sys
from pathlib import Path


def prepare_submission(input_file: Path, output_file: Path):
    """Convert normalized predictions to submission format."""
    
    submission_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            submission_entry = {
                'ipis_id': entry['ipis_id'],
                'target': entry['target'],
                'generated_target': entry['target'],
                'normalised_target': entry['normalised_target']
            }
            
            submission_entries.append(submission_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in submission_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Processed {len(submission_entries)} entries")
    print(f"Submission file saved to: {output_file}")
    print(f"\nSample entry:")
    print(json.dumps(submission_entries[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python prepare_submission.py <input.jsonl> <output.tsv>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    prepare_submission(input_file, output_file)
