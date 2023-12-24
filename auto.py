import argparse
import os
import subprocess
from shutil import move
import re

def natural_keys(text):
    """Helper function for natural sorting (sorts text with embedded numbers correctly)."""
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]

def list_in_files_recursive(directory):
    """List all .in files recursively in the given directory."""
    in_files = {}
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.in'):
                key = os.path.splitext(f)[0]
                in_files[key] = os.path.join(root, f)
    return in_files

def run_gprMax(file_path, n, use_gpu):
    """Run gprMax command on a file."""
    gpu_flag = '--gpu' if use_gpu else ''
    command = f'python -m gprMax {file_path} -n {n} {gpu_flag}'
    print("Running command: ", command)
    subprocess.run(command, shell=True)

def merge_output_files(directory, file_without_ext):
    """Merge output files in the directory based on the .in filename."""
    subprocess.run(f'python -m tools.outputfiles_merge {directory}/{file_without_ext}', shell=True)

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_output_file(source_path, dest_directory):
    """Move the file to the specified output directory, creating the directory if necessary."""
    ensure_directory_exists(dest_directory)
    move(source_path, dest_directory)

def main(args):
    in_files = list_in_files_recursive(args.input)
    sorted_keys = sorted(in_files.keys(), key=natural_keys)

    # Determine start and end indices
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(sorted_keys)

    for key in sorted_keys[start:end]:
        file_path = in_files[key]
        run_gprMax(file_path, args.n, args.gpu)

        if args.merge:
            merge_output_files(os.path.dirname(file_path), key)
            merged_file = key + '_merged.out'
            merged_path = os.path.join(os.path.dirname(file_path), merged_file)
            move_output_file(merged_path, os.path.join(args.output, merged_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .in files with gprMax.')
    parser.add_argument('-i', '--input', required=True, help='Input folder path')
    parser.add_argument('--start', type=int, help='Start index (optional)')
    parser.add_argument('--end', type=int, help='End index (optional)')
    parser.add_argument('--merge', action='store_true', default=False, help='Merge output files')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for processing')
    parser.add_argument('-n', required=True, type=int, help='Number of iterations')
    parser.add_argument('-o', '--output', required=True, help='Output folder path')
    args = parser.parse_args()

    main(args)
