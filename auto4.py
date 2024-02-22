import argparse
import os
import subprocess
from shutil import move
import re
import matplotlib.pyplot as plt
from tools.plot_Bscan import get_output_data, mpl_plot
from PIL import Image

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
    gpu_flag = '-gpu' if use_gpu else ''
    command = f'python -m gprMax {file_path} -n {n} {gpu_flag}'
    print("Running command: ", command)
    subprocess.run(command, shell=True, check=True)

def merge_output_files(directory, file_without_ext):
    """Merge output files in the directory based on the .in filename and delete the original .out files."""
    subprocess.run(f'python -m tools.outputfiles_merge {directory}/{file_without_ext}', shell=True, check=True)
    
    # New code to delete original .out files
    pattern = f'^{file_without_ext}\d+\.out$'
    for f in os.listdir(directory):
        if re.match(pattern, f):
            os.remove(os.path.join(directory, f))

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_output_file(source_path, dest_directory):
    """Move the file to the specified output directory, creating the directory if necessary."""
    ensure_directory_exists(dest_directory)
    dest_path = os.path.join(dest_directory, os.path.basename(source_path))
    move(source_path, dest_path)

    return dest_path

def process_file(file_path, rxnumber, rxcomponent, non_greyscale_dir, greyscale_dir):
    """Generate plots for a .out file, save color and grayscale images."""
    outputdata, dt = get_output_data(file_path, rxnumber, rxcomponent)
    plt_figure = mpl_plot(file_path, outputdata, dt, rxnumber, rxcomponent)
    plt_figure.axis('off')

    savefile = os.path.splitext(os.path.basename(file_path))[0]
    image_path_with_colorbar = os.path.join(non_greyscale_dir, savefile + '.jpg')
    grayscale_image_path = os.path.join(greyscale_dir, savefile + '_grayscale.jpg')

    plt_figure.savefig(image_path_with_colorbar, dpi=150, format='JPEG', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    color_bar_width = 275
    image = Image.open(image_path_with_colorbar)
    cropped_image = image.crop((0, 0, image.width - color_bar_width, image.height))
    grayscale_image = cropped_image.convert('L')
    grayscale_image.save(grayscale_image_path)

def main(args):
    in_files = list_in_files_recursive(args.input)
    sorted_keys = sorted(in_files.keys(), key=natural_keys)

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(sorted_keys)

    processed_count = 0
    processed_files = []  # Keep track of processed _merged.out files

    rxnumber = 1
    rxcomponent = 'Ez'
    non_greyscale_dir = 'images/non'
    greyscale_dir = 'images/greyscale'
    ensure_directory_exists(non_greyscale_dir)
    ensure_directory_exists(greyscale_dir)

    for key in sorted_keys[start:end]:
        file_path = in_files[key]
        run_gprMax(file_path, args.n, args.gpu)

        if args.merge:
            merge_output_files(os.path.dirname(file_path), key)
            merged_file = key + '_merged.out'
            # Adjust merged_path to use the output directory where the file has been moved
            merged_path = os.path.join(os.path.dirname(file_path), merged_file)  # Adjusted to reflect the correct directory
            new_path = move_output_file(merged_path, args.output)  # This operation may be redundant if merged_path already points to the correct location

            process_file(new_path, rxnumber, rxcomponent, non_greyscale_dir, greyscale_dir)
            processed_files.append(new_path)
            processed_count += 1

    # After all files processed, check for any remaining _merged.out files to process
    if processed_files:
        for file in processed_files:
            os.remove(file)

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
