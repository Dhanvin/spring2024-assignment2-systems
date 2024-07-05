import subprocess

# Define the parameter sets
parameter_sets = [
    (768, 3072, 12, 12),
    (1024, 4096, 24, 16),
    (1280, 5120, 36, 20),
    (1600, 6400, 48, 25),
    (2560, 10240, 32, 32)
]

# Path to your benchmark.py script
benchmark_script = 'path/to/benchmark.py'

# Iterate over parameter sets and execute benchmark.py
for d_model, d_ff, num_layers, num_heads in parameter_sets:
    command = [
        'python', benchmark_script,
        '--d_model', str(d_model),
        '--d_ff', str(d_ff),
        '--num_layers', str(num_layers),
        '--num_heads', str(num_heads)
    ]
    
    # Execute the command
    subprocess.run(command)

    # Add newline for clarity between different runs
    print("\n")