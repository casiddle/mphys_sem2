
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

data_file_path="Data/full_data_set_sem1.csv"
def theta_to_r(theta, distance):
    
    r=distance*np.tan(theta)
    return r

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

# Load into a DataFrame
df = pd.read_csv(data_file_path)
df['R'] = df['Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))
# Separate features and target
X = df[["UV/X-ray", "R", "Critical Energy"]].values
y = df["Emittance"].values

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape to match model output



# Create dataset and dataloader
dataset = EmittanceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Batch size = 2

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a scaler
scaler = StandardScaler()
# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Use the same scaler (don't refit) to scale the test data
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data into tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# Create training dataset and dataloader
train_dataset = EmittanceDataset(X_train_tensor, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Create test dataset and dataloader
test_dataset = EmittanceDataset(X_test_tensor, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

for batch_features, batch_targets in train_dataloader:
    print("Features:", batch_features)
    print("Targets:", batch_targets)
    break  # Check one batch

# Check if a GPU (CUDA) is available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using {device} device")

class NeuralNetwork(nn.Module): #define custom neural network
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

print(model)   

# Step 1: Define a loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate = 0.001

# Step 2: Training Loop
epochs = 120  # Number of epochs to train
epoch_array=np.empty(0)
loss_array=np.empty(0)
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    
    # Loop over batches of data
    for batch_features, batch_targets in train_dataloader:
        # Move data to the device (GPU or CPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Step 3: Forward pass (make predictions)
        predictions = model(batch_features)
        
        # Step 4: Calculate the loss
        loss = loss_fn(predictions, batch_targets)
        
        # Step 5: Backpropagation (compute gradients)
        loss.backward()
        
        # Step 6: Update the weights using the optimizer
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
    
    # Print loss for every epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}")
    epoch_array=np.append(epoch_array,epoch)
    loss_array=np.append(loss_array,running_loss/len(dataloader))

# After training is done, plot the results
plt.figure(figsize=(10, 6))
plt.plot(epoch_array, loss_array, label='Training Loss', color='tab:blue', marker='o')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss vs Epoch', fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig(f'Machine Learning/Plots/Epochs_vs_loss_{epochs}_epochs.png', dpi=250)
plt.show()

#focus on the last epochs
no_epochs_focus=40

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
plt.savefig(f'Machine Learning/Plots/Last_{no_epochs_focus}_Epochs_vs_loss_{epochs}_epochs.png', dpi=250)

plt.show()




# Step 3: Evaluate the model
# Evaluate the model on a test dataset
# Assuming you have a test DataLoader called `test_dataloader`

# First, set the model to evaluation mode (turns off dropout, batchnorm, etc.)
model.eval()

# Initialize variables to keep track of the total loss and number of samples
total_loss = 0.0
num_samples = 0

# No need to calculate gradients during evaluation, so we use torch.no_grad()
with torch.no_grad():
    for batch_features, batch_targets in test_dataloader:
        # Move the data to the appropriate device (CPU or GPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Make predictions
        predictions = model(batch_features)
        
        # Calculate the loss (using the same loss function as in training)
        loss = loss_fn(predictions, batch_targets)
        
        # Accumulate the loss and number of samples
        total_loss += loss.item() * batch_features.size(0)  # Multiply by batch size
        num_samples += batch_features.size(0)

# Calculate average loss
average_loss = total_loss / num_samples
print(f"Test Set Loss: {average_loss:.4f}")

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

# Print or store your predictions (e.g., for comparison with actual targets)
#print("Predictions on the Test Data:", all_predictions)

# If you want to compare with the true values (targets) on the test set
test_targets = np.concatenate([batch_targets.cpu().numpy() for _, batch_targets in test_dataloader], axis=0)
#print("True Values (Emittance):", test_targets)



mse=average_loss
#all_predictions = all_predictions.numpy()
y_test = y_test.numpy()

x=np.linspace(min(min(test_predictions), min(test_targets)),max(max(test_predictions), max(test_targets)),100)
x = x.flatten() # Convert to 1D array
print("x:",x)
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
ax2.errorbar(test_targets, residuals, color='tab:blue', alpha=0.7, fmt='o',label="Residuals")

ax2.axhline(0, color='k', linestyle='--', linewidth=1)
ax2.axhline(-1,color='r',linestyle='--',linewidth=1)
ax2.axhline(1,color='r',linestyle='--',linewidth=1)
ax2.set_ylabel(r"Residuals ($\sigma$)", fontsize=14)
ax2.set_xlabel(r"QV3D data values for emittance ($\mu m$)", fontsize=14)
ax2.set_ylim(-np.max(np.abs(1.1*residuals/np.sqrt(mse))), np.max((np.abs(1.1*residuals/np.sqrt(mse)))))

plt.savefig(r'Machine Learning\Plots\Initial_NN_plot',dpi=250)
plt.show()