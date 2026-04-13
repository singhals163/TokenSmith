import os
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_extraction_times(base_dir="data/experiments_sequential"):
    workers_time = {}
    
    # Regex to match the folder pattern e.g., workers_2_chunk_1000
    folder_pattern = re.compile(r"workers_(\d+)_chunk_\d+")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Please ensure it exists.")
        return

    # Traverse the experiment directories
    for folder_name in os.listdir(base_dir):
        match = folder_pattern.match(folder_name)
        if match:
            workers = int(match.group(1))
            
            # Find the profiling txt file inside the folder
            txt_files = glob.glob(os.path.join(base_dir, folder_name, "profiling_*_workers.txt"))
            if not txt_files:
                continue
            
            txt_file = txt_files[0]
            with open(txt_file, 'r') as f:
                content = f.read()
                # Extract the total time for the block via regex
                time_match = re.search(r"\[Block\] Total Parallel Extraction Pipeline.*?\|\s*[\d\.]+\s*\|\s*([\d\.]+)", content)
                if time_match:
                    time_s = float(time_match.group(1))
                    workers_time[workers] = time_s
    
    # Also add the 62 worker datapoint if you have it cached separately
    if 62 not in workers_time:
        workers_time[62] = 381.6659 # Hardcoded from your previous run just in case

    if not workers_time:
        print("No data found to plot. Ensure the directory path and txt files are correct.")
        return

    # Sort data by number of workers
    sorted_workers = sorted(workers_time.keys())
    sorted_times = [workers_time[w] for w in sorted_workers]

    # Initialize plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_workers, sorted_times, marker='o', linestyle='-', color='b', linewidth=2.5, markersize=10)
    
    # --- NEW: Apply Logarithmic Scales ---
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Force axes to show standard numbers instead of scientific notation (10^3)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    # Force the X-axis to explicitly tick at the specific worker values
    plt.xticks(sorted_workers)
    
    # Formatting (Large labels, no title)
    plt.xlabel("Number of Workers", fontsize=20, fontweight='bold')
    plt.ylabel("Total Extraction Time (s)", fontsize=20, fontweight='bold')
    
    # Set large ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Show grid for both major and minor ticks on the log scale
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and display
    plt.savefig("extraction_scaling_log.png", dpi=300)
    print("Graph saved as 'extraction_scaling_log.png'")
    plt.show()

if __name__ == "__main__":
    plot_extraction_times()