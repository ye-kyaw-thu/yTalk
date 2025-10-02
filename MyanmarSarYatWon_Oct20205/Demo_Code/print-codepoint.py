#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For printing code points (decimal, unicode) of each Myanmar character and counting total no. of characters
Written by Ye Kyaw Thu, Visiting Professor,
Language Semantic Technology Research Team (LST), NECTEC, Thailand

How to run:
    python print-codepoint.py --input filename
    python print-codepoint.py "word1" "word2"
    python print-codepoint.py "လေယာဉ်ပျံ" "လေယာဥ်ပျံ"
"""

import argparse
import sys
import os

def print_codepoints(text, output_file=None):
    """
    Print code points for each character in the text
    """
    if output_file:
        output_file.write(f"{text}\n")
        print(text)
    else:
        print(text)
    
    # Remove newline for processing but keep original for display
    processed_text = text.rstrip('\n\r')
    
    results = []
    for char in processed_text:
        decimal_code = ord(char)
        unicode_hex = f"U{decimal_code:04X}"
        results.append(f"{char} ({decimal_code}, {unicode_hex})")
    
    output_line = " ".join(results) + f", no. of char = {len(processed_text)}"
    
    if output_file:
        output_file.write(output_line + "\n")
        output_file.flush()
    print(output_line)

def process_file(input_file, output_file=None):
    """
    Process an input file line by line
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                print_codepoints(line, output_file)
                if output_file:
                    output_file.write("\n")
                print()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

def process_words(words, output_file=None):
    """
    Process words provided as command line arguments
    """
    for word in words:
        print_codepoints(word, output_file)
        if output_file and word != words[-1]:
            output_file.write("\n")
        if word != words[-1]:
            print()

def main():
    parser = argparse.ArgumentParser(
        description='Print code points (decimal, unicode) of each Myanmar character and count total characters',
        epilog='Examples:\n'
               '  python print-codepoint.py --input pair.txt\n'
               '  python print-codepoint.py "word1" "word2"\n'
               '  python print-codepoint.py "လေယာဉ်ပျံ" "လေယာဥ်ပျံ"\n'
               '  python print-codepoint.py --input pair.txt --output result.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'words', 
        nargs='*',
        help='Words to process (optional if --input is provided)'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input file containing text to process'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.words:
        parser.print_help()
        print("\nError: Either provide an input file with --input or words as arguments", file=sys.stderr)
        sys.exit(1)
    
    if args.input and args.words:
        print("Warning: Both input file and words provided. Processing input file only.", file=sys.stderr)
    
    # Setup output
    output_file = None
    if args.output:
        try:
            output_file = open(args.output, 'w', encoding='utf-8')
        except Exception as e:
            print(f"Error creating output file: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Process input
        if args.input:
            process_file(args.input, output_file)
        else:
            process_words(args.words, output_file)
    
    finally:
        # Close output file if opened
        if output_file:
            output_file.close()
            print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()

