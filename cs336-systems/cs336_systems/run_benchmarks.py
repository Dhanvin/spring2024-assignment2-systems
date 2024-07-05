import subprocess

# Define the parameter sets
parameter_sets = [
    (768, 12, 12),
    (1024, 24, 16),
    (1280, 36, 20),
    (1600, 48, 25),
    (2560, 32, 32)
]

# Path to your benchmark.py script
benchmark_script = '/content/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark_lm.py'

# Iterate over parameter sets and execute benchmark.py
print("Benchmarking forward pass")
for d_model, num_layers, num_heads in parameter_sets:
    command = [
        'python', benchmark_script,
        '--d_model', str(d_model),
        '--num_layers', str(num_layers),
        '--num_heads', str(num_heads),
        '--mode', 'forward'
    ]
    
    print(command)

    # Execute the command
    subprocess.run(command)

    print("\n")

  
# Iterate over parameter sets and execute benchmark.py
print("Benchmarking full pass")
for d_model, num_layers, num_heads in parameter_sets:
    command = [
        'python', benchmark_script,
        '--d_model', str(d_model),
        '--num_layers', str(num_layers),
        '--num_heads', str(num_heads),
        '--mode', 'full'
    ]
    print(command)

    # Execute the command
    subprocess.run(command)

    print("\n")
