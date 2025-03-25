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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import time
import os
import random
import hiddenlayer as hl

# Key inputs
save_metrics = True  # Change this to False if you don’t want to save for this run
csv_file_path = "Machine Learning/training_metrics_3_targets_new.csv"
data_file_path="Processed_Data/data_sets/big_scan_correct.csv"

epochs = 1000  # Number of epochs to train
patience = 50  # number of epochs with no improvement before stopping
batch_no=30 #batch size
no_hidden_layers=10#number of hidden layers 
learning_rate=0.001 #learning rate
no_nodes=36 #number of nodes in each hidden layer
combine_size_val=0.2 
test_size_val=combine_size_val/2
dropout=0.2


predicted_feature=["Emittance",'Beam Energy','Beam Spread'] #name of the features to be predicted
predictor_feature=["X-ray Mean Radiation Radius",'X-ray Critical Energy', 'X-ray Percentage']
input_size=len(predictor_feature)#number of input features
activation_function="Leaky ReLU" #activation function- note this string needs to be changed manually
threshold=0.1 #percentage threshold for which a prediction can be considered accurate
#train_test_seed=42
train_test_seed=random.randint(1,100)
print("Train-test split seed:",train_test_seed)

# Define the neural network class and relative loss and optimiser functions
class NeuralNetwork(nn.Module):  # Define custom neural network
    def __init__(self, input_size=input_size, hidden_size=no_nodes, num_hidden_layers=no_hidden_layers, num_outputs=len(predicted_feature),dropout_rate=dropout):
        super().__init__()
        
        # Initialize an empty list to hold layers
        layers = []
        
        # First hidden layer (input layer to the first hidden layer)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size)) #Normalisation
        #layers.append(nn.ReLU())  # ReLU activation function
        layers.append(nn.LeakyReLU(negative_slope=0.01))  
        
        # Loop to add the hidden layers
        for _ in range(num_hidden_layers - 1):  # Subtract 1 since the first hidden layer is already added
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size)) 
            #layers.append(nn.ReLU())  # ReLU activation for each hidden layer
            layers.append(nn.LeakyReLU(negative_slope=0.01))  
            layers.append(nn.Dropout(dropout_rate))  # Dropout layer after activation
        
        # Output layer
        layers.append(nn.Linear(hidden_size, num_outputs))  # Output layer (single output)
        #layers.append(nn.ReLU())
        layers.append(nn.LeakyReLU(negative_slope=0.001))  
        
        # Use Sequential to combine layers
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits




# Check if a GPU (CUDA) is available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using {device} device")

model = NeuralNetwork(input_size=input_size, hidden_size=no_nodes, num_hidden_layers=no_hidden_layers, num_outputs=len(predicted_feature), dropout_rate=dropout).to(device)  # Initialize the model
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


def mse_cal(predictions,actual):
    length=len(predictions)
    mse=(1/length)*np.sum((predictions-actual)**2)
    return mse

# MAIN-------------------------------------------------------------------



df = pd.read_csv(data_file_path)
data_size=len(df)
print("Total number of data points:",data_size)
df['X-ray Mean Radiation Radius'] = df['X-ray Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))
df['UV Percentage']=df['No. UV Photons']/df['Total no. Photons']
df['Other Percentage']=df['No. Other Photons']/df['Total no. Photons']


print("Total number of data points:",data_size)
data_size=len(df)
# Separate features and target
X = df[predictor_feature].values # Features
y = df[predicted_feature].values # Target



y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)



# Convert to PyTorch tensors
X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(y_scaled, dtype=torch.float32).reshape(data_size, 3)  # Reshape to match model output



# Create dataset and dataloader
dataset = EmittanceDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_no, shuffle=True)  # Batch size = 2

# Split data into training and test sets (80% train, 20% test)
X_train, X_combine, y_train, y_combine = train_test_split(X, y, test_size=combine_size_val, random_state=train_test_seed)
X_test,X_validation,y_test,y_validation= train_test_split(X_combine, y_combine, test_size=int(combine_size_val/test_size_val), random_state=train_test_seed)



# Convert the data into tensors
X_train_tensor = X_train.clone().detach().float()
X_test_tensor = X_test.clone().detach().float()
X_validation_tensor = X_validation.clone().detach().float()

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

#Create validation dataset and dataloader
validation_dataset=EmittanceDataset(X_validation_tensor, y_validation)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_no, shuffle=False)





#Training Loop
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
        #print(f"Batch size: {batch_features.size(0)}")
        # Move data to the device (GPU or CPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        optimizer.zero_grad() #check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        predictions = model(batch_features)
        loss = loss_fn(predictions, batch_targets)
        loss.backward() #check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        optimizer.step()
        running_loss += loss.item()
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)


    epoch_array=np.append(epoch_array,epoch)
    loss_array=np.append(loss_array,running_loss/len(train_dataloader))
    # Validation step
    model.eval()  
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for efficiency
        for val_features, val_targets in validation_dataloader:
            val_features, val_targets = val_features.to(device), val_targets.to(device)
            val_predictions = model(val_features)
            val_loss += loss_fn(val_predictions, val_targets).item()

    avg_train_loss = running_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(validation_dataloader)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Early stopping logic
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement >= patience:
        print("Early stopping triggered!")
        break
# End overall training timer
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
# Print average and total training time
avg_epoch_time = np.mean(epoch_times)
print(f"Average time per epoch: {avg_epoch_time:.2f} sec")
print(f"Total training time: {total_training_time:.2f} sec")


# After training is done, plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(epoch_array, loss_array, label='Training Loss', color='tab:blue', marker='o')
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.title('Training Loss vs Epoch', fontsize=14)
# plt.grid(True)
# plt.legend()
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
# plt.figure(figsize=(10, 6))
# plt.plot(epoch_array_last, loss_array_last, label='Training Loss', color='tab:blue', marker='o')
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.title(f'Training Loss vs Epoch (Last {no_epochs_focus} Epochs)', fontsize=14)
# plt.grid(True)
# plt.legend()
#plt.savefig(f'Machine Learning/Plots/Last_{no_epochs_focus}_Epochs_vs_loss_{epochs}_epochs.png', dpi=250)
#plt.show()

# Evaluate the model on a test dataset to check its performance___________________________________________________

# First, set the model to evaluation mode (turns off dropout, batchnorm, etc.)--- this step very important
model.eval()

# Initialize variables to keep track of the total loss and number of samples
total_loss = 0.0
num_samples = 0
loss_array=np.empty(0)
correct_predictions=np.zeros(len(predicted_feature))  # One for each feature


# Array to store individual losses for each feature
individual_loss_array = np.zeros(len(predicted_feature)) 


# No need to calculate gradients during evaluation, so we use torch.no_grad()
with torch.no_grad():
    for batch_features, batch_targets in test_dataloader:
        # Move the data to the appropriate device (CPU or GPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Make predictions
        predictions = model(batch_features)

         # Inverse scaling: Transform predictions and batch_targets back to the original scale
        predictions = y_scaler.inverse_transform(predictions.cpu().numpy())
        batch_targets = y_scaler.inverse_transform(batch_targets.cpu().numpy())

        # Convert predictions and batch_targets back to PyTorch tensors
        predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device)

        # For each feature, calculate the percentage error
        for i in range(predictions.shape[1]):  # Loop over each feature (output)
            #print("Predictions:",predictions[:, i])
            #print("Target:",batch_targets[:, i])
            percentage_error = torch.abs(predictions[:, i] - batch_targets[:, i]) / batch_targets[:, i]
            correct_predictions[i] += (percentage_error <= threshold).sum().item()  # Count predictions within threshold

        # Calculate individual losses for each output
        individual_losses = [loss_fn(predictions[:, i], batch_targets[:, i]) for i in range(predictions.shape[1])]
        
        # Store individual losses
        for i, loss in enumerate(individual_losses):
            individual_loss_array[i] += loss.item() * batch_features.size(0)  # Scale by batch size
        
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
mse=average_loss
mean_individual_losses = individual_loss_array / num_samples
# Calculate accuracy
# Calculate accuracy for each feature
accuracy_per_feature = correct_predictions / num_samples

# Display Results
# Display accuracy for each feature
for i, feature in enumerate(predicted_feature):
    print(f"Accuracy for {feature} within {threshold*100:.2f}%: {accuracy_per_feature[i] * 100:.2f}%")

print(f"Overall Test Loss: {average_loss:.4f}")
for i, feature in enumerate(predicted_feature):
    print(f"{feature} Test Loss: {mean_individual_losses[i]:.4f}")

#Evaluate how model performs on data it was trained on
# Initialize variables to keep track of the total loss and number of samples
total_loss_train = 0.0
num_samples_train = 0
loss_array_train=np.empty(0)

# Array to store individual losses for each feature
individual_loss_array_train = np.zeros(len(predicted_feature)) 

# No need to calculate gradients during evaluation, so we use torch.no_grad()
with torch.no_grad():
    for batch_features, batch_targets in train_dataloader:
        # Move the data to the appropriate device (CPU or GPU)
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        
        # Make predictions
        predictions = model(batch_features)

        # Inverse scaling: Transform predictions and batch_targets back to the original scale
        predictions = y_scaler.inverse_transform(predictions.cpu().numpy())
        batch_targets = y_scaler.inverse_transform(batch_targets.cpu().numpy())

        # Convert predictions and batch_targets back to PyTorch tensors
        predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device)

        # Calculate individual losses for each output
        individual_losses_train = [loss_fn(predictions[:, i], batch_targets[:, i]) for i in range(predictions.shape[1])]
        
        # Store individual losses
        for i, loss in enumerate(individual_losses_train):
            individual_loss_array_train[i] += loss.item() * batch_features.size(0)  # Scale by batch size

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
mse=average_loss
mean_individual_losses_train = individual_loss_array_train / num_samples
# Display Results
print(f"Overall Train Loss: {average_loss_train:.4f}")
for i, feature in enumerate(predicted_feature):
    print(f"{feature} Train Loss: {mean_individual_losses_train[i]:.4f}")

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
    'test_loss_error': loss_error,
    'optimiser': type(optimizer).__name__,
    'learning_rate': learning_rate,
    'activation_function': activation_function,
    'no_hidden_layers': no_hidden_layers ,
    'batch_size': batch_no,
    'no_nodes': no_nodes,
    'predicted_feature': predicted_feature,
    'predictor_features': predictor_feature,  
    'training_loss': average_loss_train,
    'overfitting':overfitting,
    'test_size':test_size_val,
    'dropout_rate':dropout,
    'accuracy_threshold':threshold
}

# Add individual feature losses
for i, feature in enumerate(predicted_feature):
    metrics[f'{feature}_train_loss'] = mean_individual_losses_train[i]
    metrics[f'{feature}_test_loss'] = mean_individual_losses[i]
    metrics[f'{feature}_test_accuracy']=accuracy_per_feature[i]

# Print the full metrics dictionary 
print("Metrics Dictionary:", metrics)


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


# Save model state dictionary
torch.save(model.state_dict(), r'Machine Learning\models\model1.pth')
#______________________________________________________________________________________________________________________________________________________________________


model.eval()


# Extract all test data in one go
all_features, all_targets = next(iter(test_dataloader))

# Move data to the correct device
all_features = all_features.to(device)
all_targets = all_targets.to(device)

# Disable gradient calculation for efficiency
with torch.no_grad():
    predictions = model(all_features).cpu().numpy()
    # Inverse scaling: Transform predictions and batch_targets back to the original scale
    predictions = y_scaler.inverse_transform(predictions)
    actual_values =y_scaler.inverse_transform(all_targets.cpu().numpy())


beam_spread_preds = predictions[:, 2]
beam_energy_preds = predictions[:, 1]
emittance_preds = predictions[:, 0]

beam_spread_actuals = actual_values[:, 2]
beam_energy_actuals = actual_values[:, 1]
emittance_actuals = actual_values[:, 0]





print("Emittance predictions:",emittance_preds)
print("Actual Emittance:",emittance_actuals)
length=len(emittance_preds)
print("Length:", length)

em_mse=mse_cal(emittance_preds,emittance_actuals)
energy_mse=mse_cal(beam_energy_preds,beam_energy_actuals)
spread_mse=mse_cal(beam_spread_preds,beam_spread_actuals)
print("My calculated emittance mse:",em_mse)
print("My calculated beam energy mse:",energy_mse)
print("My calculated beam spread mse:",spread_mse)

#print("Beam spread predictions:",beam_spread_preds)
#print("Actual Beam spread:",beam_spread_actuals)


# Calculate residuals
beam_spread_residuals = beam_spread_actuals - beam_spread_preds
beam_energy_residuals = beam_energy_actuals - beam_energy_preds
emittance_residuals = emittance_actuals - emittance_preds

# Plotting
plt.figure(figsize=(15, 10))

# Scatter plots
plt.subplot(2, 3, 1)
plt.scatter(beam_spread_actuals, beam_spread_preds, color='blue', alpha=0.6)
plt.plot(beam_spread_actuals, beam_spread_actuals, color='k', linestyle='-')
plt.title('Beam Spread: Actual vs Predicted')
plt.xlabel('Actual Beam Spread')
plt.ylabel('Predicted Beam Spread')

# Residual plots
plt.subplot(2, 3, 4)
plt.scatter(beam_spread_actuals, beam_spread_residuals, color='blue', alpha=0.6)
plt.axhline(0, color='k', linestyle='--')
plt.title('Beam Spread Residuals')
plt.xlabel('Actual Beam Spread')
plt.ylabel('Residual')

# Scatter plots
plt.subplot(2, 3, 2)
plt.scatter(beam_energy_actuals, beam_energy_preds, color='green', alpha=0.6)
plt.plot(beam_energy_actuals, beam_energy_actuals, color='k', linestyle='-')
plt.title('Beam Energy: Actual vs Predicted')
plt.xlabel('Actual Beam Energy')
plt.ylabel('Predicted Beam Energy')

# Residual plots
plt.subplot(2, 3, 5)
plt.scatter(beam_energy_actuals, beam_energy_residuals, color='green', alpha=0.6)
plt.axhline(0, color='k', linestyle='--')
plt.title('Beam Energy Residuals')
plt.xlabel('Actual Beam Energy')
plt.ylabel('Residual')

# Scatter plots
plt.subplot(2, 3, 3)
plt.scatter(emittance_actuals, emittance_preds, color='purple', alpha=0.6)
plt.plot(emittance_actuals, emittance_actuals, color='k', linestyle='-')
plt.title('Emittance: Actual vs Predicted')
plt.xlabel('Actual Emittance')
plt.ylabel('Predicted Emittance')

# Residual plots
plt.subplot(2, 3, 6)
plt.scatter(emittance_actuals, emittance_residuals, color='purple', alpha=0.6)
plt.axhline(0, color='k', linestyle='--')
plt.title('Emittance Residuals')
plt.xlabel('Actual Emittance')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()