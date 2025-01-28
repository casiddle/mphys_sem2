
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
epochs = 100  # Number of epochs to train
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
all_predictions = np.concatenate(all_predictions, axis=0)

# Print or store your predictions (e.g., for comparison with actual targets)
#print("Predictions on the Test Data:", all_predictions)

# If you want to compare with the true values (targets) on the test set
test_targets = np.concatenate([batch_targets.cpu().numpy() for _, batch_targets in test_dataloader], axis=0)
#print("True Values (Emittance):", test_targets)

import matplotlib.pyplot as plt

# Assuming `all_predictions` contains the predicted emittance values
# And `test_targets` contains the actual (true) emittance values

# Create a scatter plot to visualize the predictions vs true values
plt.figure(figsize=(8, 6))
plt.scatter(test_targets, all_predictions, color='blue', alpha=0.5)

# Adding labels and title
plt.xlabel("Actual Emittance")
plt.ylabel("Predicted Emittance")
plt.title("Actual vs Predicted Emittance")

# Optionally, add a line for perfect predictions (y = x line)
plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], color='red', linestyle='--', label="Perfect Prediction")

# Display the plot
plt.legend()
plt.show()

