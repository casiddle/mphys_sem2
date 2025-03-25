import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

def extract_properties(output):
    """
    Extract properties from the output string of the synchrotron_carys3D_sys_input.py script.
    This function assumes properties are on separate lines in a specific format.
    """
    properties = {}
    lines = output.strip().split('\n')
    
    for line in lines:
        try:
            # Adjust based on actual output format
            if "Overall Test Loss:" in line:
                properties['overall_test_loss'] = float(line.split(":")[-1].strip().split()[0])  # Extract the first part after ':'
            elif "Emittance Test Loss:" in line:
                properties['emittance_test_loss'] = float(line.split(":")[-1].strip())
            elif "Beam Energy Test Loss:" in line:
                properties['beam_energy_test_loss'] = float(line.split(":")[-1].strip())
            elif "Beam Spread Test Loss:" in line:
                properties['beam_spread_test_loss'] = float(line.split(":")[-1].strip())

          
        except ValueError:
            print(f"Could not convert the line to float: '{line}'")
    
    return properties

def get_properties( no_hidden_layers, no_nodes, learning_rate):
    """
    Run neural_network.py with given parameters and return the extracted properties.
    """
    cmd = [
        "python", r"Machine Learning\hyperparameter_tuning\neural_network.py",
        "--no_hidden_layers", str(no_hidden_layers),
        "--no_nodes", str(no_nodes),
        "--learning_rate", str(learning_rate)
    ]

    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        #print("Standard Output:", result.stdout)
        #print("Error Output:", result.stderr)
        
        # Return the extracted properties from the output
        return extract_properties(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error while processing")
        print("Error Output:", e.stderr)
        print("Standard Output:", e.stdout)
        return None
    



# List of hyperparameters to tune
hidden_layers_list = np.round(np.linspace(10,100,25),0)  
nodes_list = np.round(np.linspace(12,36,5),0)  
learning_rate_list = [0.01]  

# Store results
results = []

# Loop over combinations of hyperparameters
for no_hidden_layers in hidden_layers_list:
    for no_nodes in nodes_list:
        for learning_rate in learning_rate_list:
            print(f"Running for: Hidden Layers = {int(no_hidden_layers)}, Nodes = {no_nodes}, Learning Rate = {learning_rate}")
            properties = get_properties(int(no_hidden_layers), int(no_nodes), learning_rate)
            if properties:
                properties['no_hidden_layers'] = no_hidden_layers
                properties['no_nodes'] = no_nodes
                properties['learning_rate'] = learning_rate
                results.append(properties)

# Convert results to a pandas DataFrame for easier analysis
df = pd.DataFrame(results)

# Display the DataFrame with results
print("\nTuning Results:")
print(df)

# Find the optimal combination that minimizes each loss
optimal_combination = {
    'overall_test_loss': df.loc[df['overall_test_loss'].idxmin()],
    'emittance_test_loss': df.loc[df['emittance_test_loss'].idxmin()],
    'beam_energy_test_loss': df.loc[df['beam_energy_test_loss'].idxmin()],
    'beam_spread_test_loss': df.loc[df['beam_spread_test_loss'].idxmin()]
}

# Display optimal combinations
print("\nOptimal Combinations:")
for loss, row in optimal_combination.items():
    print(f"{loss}: {row[['no_hidden_layers', 'no_nodes', 'learning_rate', loss]]}")

# # Optionally: plot results
# df.set_index(['no_hidden_layers', 'no_nodes', 'learning_rate'], inplace=True)
# df.plot(y=['overall_test_loss', 'emittance_test_loss', 'beam_energy_test_loss', 'beam_spread_test_loss'], 
#         subplots=True, layout=(2, 2), figsize=(10, 10))
# plt.show()