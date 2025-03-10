import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os
import random

# Key inputs
save_metrics = True  # Change this to False if you don’t want to save for this run
csv_file_path = "Machine Learning/training_metrics_3_targets.csv"
data_file_path="Processed_Data/data_sets/output_test_96.csv"

epochs = 420  # Number of epochs to train
patience = 20  # number of epochs with no improvement before stopping
batch_no=13 #batch size
no_hidden_layers=13 #number of hidden layers 
learning_rate=0.001 #learning rate
no_nodes=8 #number of nodes in each hidden layer
test_size_val=0.5 #proportion of data that is tested, 1-test_size= train_size

input_size=6 #number of input features
predicted_feature=["Emittance",'Beam Energy','Beam Spread'] #name of the features to be predicted
activation_function="ReLU" #activation function- note this string needs to be changed manually

#train_test_seed=42
train_test_seed=random.randint(1,100)
print("Train-test split seed:",train_test_seed)

# Define the neural network class and relative loss and optimiser functions
class NeuralNetwork(nn.Module):  # Define custom neural network
    def __init__(self, input_size=3, hidden_size=10, num_hidden_layers=3, num_outputs=len(predicted_feature)):
        super().__init__()
        
        # Initialize an empty list to hold layers
        layers = []
        
        # First hidden layer (input layer to the first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())  # ReLU activation function
        
        # Loop to add the hidden layers
        for _ in range(num_hidden_layers - 1):  # Subtract 1 since the first hidden layer is already added
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU activation for each hidden layer
        
        # Output layer
        layers.append(nn.Linear(hidden_size, num_outputs))  # Output layer (single output)
        
        # Use Sequential to combine layers
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Check if a GPU (CUDA) is available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using {device} device")

model = NeuralNetwork(input_size=input_size, hidden_size=no_nodes, num_hidden_layers=no_hidden_layers).to(device)  # Initialize the model
print(model)
loss_fn = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with learning rate = 0.001


class EmittanceDataset(Dataset):
    # Constructor method to initialize the dataset with features and targets
    def __init__(self, features, targets):
        # The features (input data) will be stored in self.features
        # The targets (output labels) will be stored in self.targets
        self.features = features
        self.targets = targets

    # This method defines the length of the dataset, i.e., how many samples are there
    def __len__(self):
        # Return the number of samples in the dataset (based on the features array)
        return len(self.features)

    # This method retrieves a sample from the dataset based on the index (idx)
    def __getitem__(self, idx):
        # Return the feature (input data) and the corresponding target (label)
        # at the given index `idx`
        return self.features[idx], self.targets[idx]

#Functions
def theta_to_r(theta, distance):
    
    r=distance*np.tan(theta)
    return r

# MAIN-------------------------------------------------------------------
df = pd.read_csv(data_file_path)
df['R'] = df['Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))
df['UV Percentage']=df['No. UV Photons']/df['Total no. Photons']
df['Other Percentage']=df['No. Other Photons']/df['Total no. Photons']
# Separate features and target
X = df[["R", "Critical Energy",'X-ray Critical Energy', 'X-ray Percentage','UV Percentage','Other Percentage']].values # Features
y = df[predicted_feature].values # Target

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(96, 3)  # Reshape to match model output



# Create dataset and dataloader
dataset = EmittanceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Batch size = 2

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=train_test_seed)

# Initialize a scaler
scaler = StandardScaler()
# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Use the same scaler (don't refit) to scale the test data
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data into tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

#check if there is any overlap between training and test sets
# Convert tensors to numpy for easy comparison
train_features_np = X_train_tensor.numpy()
test_features_np = X_test_tensor.numpy()

# Check if there is any overlap between training and test sets
overlap = np.isin(test_features_np, train_features_np).all(axis=1).sum()
print(f"Number of overlapping samples: {overlap}")


# Create training dataset and dataloader
train_dataset = EmittanceDataset(X_train_tensor, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_no, shuffle=True)

# Create test dataset and dataloader
test_dataset = EmittanceDataset(X_test_tensor, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_no, shuffle=False)

# for batch_features, batch_targets in train_dataloader:
#     print("Features:", batch_features)
#     print("Targets:", batch_targets)
#     break  # Check one batch





# Step 2: Training Loop
epoch_times = [] 
epochs_since_improvement = 0  # Initialize counter for early stopping
# Initialize best_loss with a very large number
best_loss = float('inf')  # Start with infinity, so any loss will be smaller

epoch_array=np.empty(0)
loss_array=np.empty(0)

total_start_time = time.time()
for epoch in range(epochs):
    start_time = time.time()
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    
    # Loop over batches of data
    for batch_features, batch_targets in train_dataloader:
        # Move data to the device (GPU or CPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Zero the gradients from the previous step
        optimizer.zero_grad() #check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # Step 3: Forward pass (make predictions)
        predictions = model(batch_features)
        
        # Step 4: Calculate the loss
        loss = loss_fn(predictions, batch_targets)
        
        # Step 5: Backpropagation (compute gradients)
        loss.backward() #check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # Step 6: Update the weights using the optimizer
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

            # Check if the loss has improved
    if running_loss < best_loss:
        best_loss = running_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement >= patience and epoch>epochs/2:
        print("Early stopping triggered!")
        break
    
    # Print loss for every epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dataloader)}")
    epoch_array=np.append(epoch_array,epoch)
    loss_array=np.append(loss_array,running_loss/len(train_dataloader))

# End overall training timer
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
# Print average and total training time
avg_epoch_time = np.mean(epoch_times)
print(f"Average time per epoch: {avg_epoch_time:.2f} sec")
print(f"Total training time: {total_training_time:.2f} sec")


# After training is done, plot the results
plt.figure(figsize=(10, 6))
plt.plot(epoch_array, loss_array, label='Training Loss', color='tab:blue', marker='o')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss vs Epoch', fontsize=14)
plt.grid(True)
plt.legend()
#plt.savefig(f'Machine Learning/Plots/Epochs_vs_loss_{epochs}_epochs.png', dpi=250)

#focus on the last epochs
no_epochs_focus=100

if len(epoch_array) > no_epochs_focus:
    epoch_array_last = epoch_array[-no_epochs_focus:]  # Slice the last no_epochs_focus epochs
    loss_array_last = loss_array[-no_epochs_focus:]    # Slice the corresponding last no_epochs_focus loss values
else:
    # If there are fewer than 20 epochs, use all epochs
    epoch_array_last = epoch_array
    loss_array_last= loss_array

# Plot the last 2no_epochs_focus epochs
plt.figure(figsize=(10, 6))
plt.plot(epoch_array_last, loss_array_last, label='Training Loss', color='tab:blue', marker='o')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Training Loss vs Epoch (Last {no_epochs_focus} Epochs)', fontsize=14)
plt.grid(True)
plt.legend()
#plt.savefig(f'Machine Learning/Plots/Last_{no_epochs_focus}_Epochs_vs_loss_{epochs}_epochs.png', dpi=250)
#plt.show()



#TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Step 3: Evaluate the model
# Evaluate the model on a test dataset to check its performance

# First, set the model to evaluation mode (turns off dropout, batchnorm, etc.)--- this step very important
model.eval()

# Initialize variables to keep track of the total loss and number of samples
total_loss = 0.0
num_samples = 0
loss_array=np.empty(0)

# No need to calculate gradients during evaluation, so we use torch.no_grad()
with torch.no_grad():
    for batch_features, batch_targets in test_dataloader:
        # Move the data to the appropriate device (CPU or GPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Make predictions
        predictions = model(batch_features)
        
        # Calculate the loss (using the same loss function as in training)
        loss = loss_fn(predictions, batch_targets)
        #print(f"Test Loss: {loss.item():.4f}")
        loss_array=np.append(loss_array,loss.item())
        
        # Accumulate the loss and number of samples
        total_loss += loss.item() * batch_features.size(0)  # Multiply by batch size
        num_samples += batch_features.size(0)

        
# Calculate average loss
average_loss = total_loss / num_samples
loss_error=np.std(loss_array)
print(f"Test Set Loss: {average_loss:.4f}")
mse=average_loss

# Initialize variables to keep track of the total loss and number of samples
total_loss_train = 0.0
num_samples_train = 0
loss_array_train=np.empty(0)

# No need to calculate gradients during evaluation, so we use torch.no_grad()
with torch.no_grad():
    for batch_features, batch_targets in train_dataloader:
        # Move the data to the appropriate device (CPU or GPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Make predictions
        predictions = model(batch_features)
        
        # Calculate the loss (using the same loss function as in training)
        loss = loss_fn(predictions, batch_targets)
        #print(f"Test Loss: {loss.item():.4f}")
        loss_array_train=np.append(loss_array,loss.item())
        
        # Accumulate the loss and number of samples
        total_loss_train += loss.item() * batch_features.size(0)  # Multiply by batch size
        num_samples_train += batch_features.size(0)

        
# Calculate average loss
average_loss_train = total_loss_train / num_samples_train
loss_error=np.std(loss_array_train)
print(f"Train Set Loss: {average_loss_train:.4f}")
mse=average_loss

if average_loss>average_loss_train:
    overfitting='Overfitting'
elif average_loss<average_loss_train:
    overfitting='Test was better??'
else:
    overfitting='Just right'
#Save necessary metrics
metrics = {
    'avg_epoch_time': avg_epoch_time,
    'total_training_time': total_training_time,
    'total_num_epochs': epochs,  # Since `epoch` is 0-indexed, add 1 to get the actual number of epochs
    'num_epochs_early_stopping': epoch+1,  # The epoch when early stopping was triggered
    'patience': patience,
    'loss_function': loss_fn,
    'test_loss': average_loss,
    'tets_loss_error': loss_error,
    'optimiser': type(optimizer).__name__,
    'learning_rate': learning_rate,
    'activation_function': activation_function,
    'no_hidden_layers': no_hidden_layers ,
    'batch_size': batch_no,
    'no_nodes': no_nodes,
    'predicted_feature': predicted_feature,  
    'training_loss': average_loss_train,
    'overfitting':overfitting,
    'test_size':test_size_val 
}

# Check if the CSV file exists (to decide whether to create or append)
if save_metrics and not os.path.exists(csv_file_path):
    # If the file doesn't exist, we need to create a new one with column headers
    columns = ['avg_epoch_time', 'total_training_time', 'total_num_epochs','num_epochs_early_stopping', 'patience', 'loss_function', 'test_loss', 'test_loss_error','optimiser', 'learning_rate', 'activation_function', 'no_hidden_layers','batch_size','no_nodes','predicted_feature','training_loss','overfitting','test_size']
    # Initialize the CSV file with column names
    training_metrics = []
else:
    # If the file exists, we will just append new data
    training_metrics = pd.read_csv(csv_file_path)

# Convert the metrics dictionary into a DataFrame
metrics_df = pd.DataFrame([metrics])

# Save the DataFrame to CSV (create if it doesn't exist, append if it does)
if save_metrics:
    if os.path.exists(csv_file_path):
        # Append new metrics to the existing CSV file
        metrics_df.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        # If CSV doesn't exist, write the DataFrame to a new CSV file with headers
        metrics_df.to_csv(csv_file_path, mode='w', header=True, index=False)

# Print the saved metrics DataFrame
print(metrics_df)


# Set model to evaluation mode (disables dropout and batch normalization layers)
model.eval()

# Initialize a list to store predictions (you could also use a tensor if preferred)
all_predictions = []

# Disable gradient calculation for predictions (this saves memory and computation)
with torch.no_grad():
    # Iterate over the test data
    for batch_features, batch_targets in test_dataloader:
        # Move the features and targets to the same device as the model
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        # Get predictions from the model
        predictions = model(batch_features)

        # Store predictions
        all_predictions.append(predictions.cpu().numpy())  # Move to CPU and convert to numpy if needed

# Convert all predictions into a single array
test_predictions = np.concatenate(all_predictions, axis=0)
test_predictions = test_predictions.flatten()  # Convert to 1D array

# Print or store your predictions (e.g., for comparison with actual targets)
#print("Predictions on the Test Data:", test_predictions)

# If you want to compare with the true values (targets) on the test set
test_targets = np.concatenate([batch_targets.cpu().numpy() for _, batch_targets in test_dataloader], axis=0)
test_targets = test_targets.flatten()  # Convert to 1D array
#print("True Values (Emittance):", test_targets)


x=np.linspace(min(min(test_predictions), min(test_targets)),max(max(test_predictions), max(test_targets)),100)
x = x.flatten() # Convert to 1D array

y_upper = x + np.sqrt(mse)
y_lower = x - np.sqrt(mse)

x_error=np.linspace(mse,mse,len(test_predictions))


# Calculate residuals
residuals =test_predictions-test_targets

# Create figure and GridSpec
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)

# Main scatter plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(test_targets, test_predictions, label="Emittance data", color='tab:blue')
ax1.plot(x, x, color='k', label=r"$y=\^y$")
ax1.fill_between(x, y_lower, y_upper, color="red", alpha=0.2, label=r"$\sqrt{\text{MSE}}$")
ax1.set_ylabel(r"Neural network model prediction for emittance ($\mu m$)", fontsize=14)
ax1.set_xlabel(r"QV3D data values for emittance ($\mu m$)", fontsize=14)
ax1.set_title(r"Emittance predicted by model vs QV3D simulation", fontsize=16)
ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', labelsize=12)

# Residuals plot (in units of sigma)
ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(test_targets, residuals/np.sqrt(mse), color='tab:blue', alpha=0.7, fmt='o',label="Residuals")

ax2.axhline(0, color='k', linestyle='--', linewidth=1)
ax2.axhline(-1,color='r',linestyle='--',linewidth=1)
ax2.axhline(1,color='r',linestyle='--',linewidth=1)
ax2.set_ylabel(r"Residuals ($\sigma$)", fontsize=14)
ax2.set_xlabel(r"QV3D data values for emittance ($\mu m$)", fontsize=14)
ax2.set_ylim(-np.max(np.abs(1.1*residuals/np.sqrt(mse))), np.max((np.abs(1.1*residuals/np.sqrt(mse)))))

#plt.savefig(r'Machine Learning\Plots\NN_plot_ReLU_3_targets',dpi=250)
#plt.show()


grad_norms = [p.grad.norm().item() for p in model.parameters()]
print("Gradient Norms:", grad_norms)


# Loop over the parameters and print them
# for name, param in model.named_parameters():
#     print(f"Parameter: {name} - Shape: {param.shape}")
#     print(param.data)  # This gives you the actual value of the weights/biases

