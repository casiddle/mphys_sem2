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
save_metrics = True # Change this to False if you don’t want to save for this run
csv_file_path = "Machine Learning/training_metrics_3_targets_2400_points.csv"
data_file_path="Processed_Data/data_sets/big_scan_2400.csv"
#torch.manual_seed(42)

emittance_loss_array=np.empty(0)
spread_loss_array=np.empty(0)
energy_loss_array=np.empty(0)

train_emittance_loss_array=np.empty(0)
train_spread_loss_array=np.empty(0)
train_energy_loss_array=np.empty(0)

# big_train_emittance_loss_array=np.empty(0)
# big_train_spread_loss_array=np.empty(0)
# big_train_energy_loss_array=np.empty(0)

learning_rate=1e-3 #learning rate
no_nodes=36 #number of nodes in each hidden layer
combine_size_val=0.2
test_size_val=combine_size_val/2
dropout=0.1
epochs = 800  # Number of epochs to train
patience = 200  # number of epochs with no improvement before stopping
batch_no=30 #batch size
no_hidden_layers=10
predicted_feature=["Emittance",'Beam Energy','Beam Spread'] #name of the features to be predicted
predictor_feature=["X-ray Mean Radiation Radius",'X-ray Critical Energy', 'X-ray Percentage']
input_size=len(predictor_feature)#number of input features
activation_function="Leaky ReLU" #activation function- note this string needs to be changed manually
threshold=0.1 #percentage threshold for which a prediction can be considered accurate
#train_test_seed=42

#print("Train-test split seed:",train_test_seed)

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
        layers.append(nn.LeakyReLU(negative_slope=0.01))  
        #layers.append(nn.Softplus())  

        
        # Use Sequential to combine layers
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class WeightedMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        weights = 1 / (y_true + 1)  # Higher weight for smaller values
        return torch.mean(weights * (y_pred - y_true) ** 2)
    
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):  # eps prevents division by zero errors
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true,eps=1e-8):
        return torch.sqrt(self.mse(y_pred, y_true) + eps)




# Check if a GPU (CUDA) is available, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using {device} device")

model = NeuralNetwork(input_size=input_size, hidden_size=no_nodes, num_hidden_layers=no_hidden_layers, num_outputs=len(predicted_feature), dropout_rate=dropout).to(device)  # Initialize the model
print(model)
loss_fn = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-10)  # Adam optimizer with learning rate 


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
    total_loss_squared=np.sum((predictions-actual)**2)
    print("Length:",length)
    #print("Total loss squared:",total_loss_squared)
    mse=(1/length)*total_loss_squared
    return mse
def get_random_rows(df, n):
    """
    This function takes a DataFrame and returns n random rows as a new DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    n (int): The number of random rows to extract.

    Returns:
    pd.DataFrame: A DataFrame containing n random rows from the original DataFrame.
    """
    # Check if n is not greater than the number of rows in df
    if n > len(df):
        raise ValueError(f"n cannot be greater than the number of rows in the DataFrame ({len(df)})")

    # Extract n random rows
    random_rows = df.sample(n)
    
    return random_rows


for i in range(0,3):
    train_test_seed=random.randint(1,300)

    
    # MAIN-------------------------------------------------------------------



    df = pd.read_csv(data_file_path)
    data_size=len(df)
    print("Total number of data points:",data_size)
    df['X-ray Mean Radiation Radius'] = df['X-ray Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))
    df['UV Percentage']=df['No. UV Photons']/df['Total no. Photons']
    df['Other Percentage']=df['No. Other Photons']/df['Total no. Photons']

    #df=df[df['Set Radius']!=0.5]




    data_size=len(df)
    print("Total number of data points:",data_size)
    # Separate features and target
    X = df[predictor_feature].values # Features
    y = df[predicted_feature].values # Target


    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)




    # Convert to PyTorch tensors
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y_scaled, dtype=torch.float32).reshape(data_size, len(predicted_feature))  # Reshape to match model output



    # Create dataset and dataloader
    #dataset = EmittanceDataset(X, y)
    #dataloader = DataLoader(dataset, batch_size=batch_no, shuffle=True)  # Batch size = 2

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_combine, y_train, y_combine = train_test_split(X, y, test_size=combine_size_val, random_state=train_test_seed)
    X_test,X_validation,y_test,y_validation= train_test_split(X_combine, y_combine, test_size=0.5, random_state=train_test_seed+1)



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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_no, shuffle=True) #shuffle should be true
    print("TRAINIG DATA LENGTH:",len(train_dataset),'-----',len(train_dataloader))

    # Create test dataset and dataloader
    test_dataset = EmittanceDataset(X_test_tensor, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_no, shuffle=False)

    #Create validation dataset and dataloader
    validation_dataset=EmittanceDataset(X_validation_tensor, y_validation)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_no, shuffle=False)
    #print(f"Validation data size: {len(X_validation)}")  # Check size of validation data
    #print("Length validation data loader:",len(validation_dataloader))





    #Training Loop
    epoch_times = [] 
    epochs_since_improvement = 0  # Initialize counter for early stopping
    # Initialize best_loss with a very large number
    best_loss = float('inf')  # Start with infinity, so any loss will be smaller

    epoch_array=np.empty(0)
    loss_array=np.empty(0)
    val_loss_array=np.empty(0)

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
        #print("Running loss:",running_loss)
        #print("Length train dataloader:",len(train_dataloader))
        #print("Running validation loss:",val_loss)
        #print("Length validation data loader",len(validation_dataloader))
        avg_val_loss = val_loss / len(validation_dataloader)
        val_loss_array=np.append(val_loss_array,avg_val_loss)

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
    print("Legth individual loss array:",len(individual_loss_array))
    print("individual losses train:",individual_losses_train)
    print("Num samples:",num_samples_train)
    mean_individual_losses_train = individual_loss_array_train / num_samples_train
    print("Mean individual losses train:",mean_individual_losses_train)
    # Display Results
    print(f"Overall Train Loss: {average_loss_train:.4f}")
    for i, feature in enumerate(predicted_feature):
        #print("i:",i)
        print(f"{feature} Train Loss: {mean_individual_losses_train[i]:.4f}")
    
    
    train_emittance_loss_array=np.append(train_emittance_loss_array,mean_individual_losses_train[0])
    train_energy_loss_array=np.append(train_energy_loss_array,mean_individual_losses_train[1])
    train_spread_loss_array=np.append(train_spread_loss_array,mean_individual_losses_train[2])

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
    #print("Metrics Dictionary:", metrics)


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
    #torch.save(model.state_dict(), r'Machine Learning\models\model1.pth')
    #______________________________________________________________________________________________________________________________________________________________________


    model.eval()


    # Extract all test data in one go
    all_features = torch.cat([batch[0] for batch in test_dataloader], dim=0).to(device)
    all_targets = torch.cat([batch[1] for batch in test_dataloader], dim=0).to(device)

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

    # # Extract all train data in one go------------------------------------------------------
    # all_features_2 = torch.cat([batch[0] for batch in train_dataloader], dim=0).to(device)
    # all_targets_2 = torch.cat([batch[1] for batch in train_dataloader], dim=0).to(device)

    # # Move data to the correct device
    # all_features_2 = all_features_2.to(device)
    # all_targets_2 = all_targets_2.to(device)

    # # Disable gradient calculation for efficiency
    # with torch.no_grad():
    #     predictions = model(all_features_2).cpu().numpy()
    #     # Inverse scaling: Transform predictions and batch_targets back to the original scale
    #     predictions = y_scaler.inverse_transform(predictions)
    #     actual_values =y_scaler.inverse_transform(all_targets_2.cpu().numpy())


    # train_beam_spread_preds = predictions[:, 2]
    # train_beam_energy_preds = predictions[:, 1]
    # train_emittance_preds = predictions[:, 0]

    # train_beam_spread_actuals = actual_values[:, 2]
    # train_beam_energy_actuals = actual_values[:, 1]
    # train_emittance_actuals = actual_values[:, 0]





    #print("Emittance predictions:",emittance_preds)
    #print("Actual Emittance:",emittance_actuals)
    length=len(emittance_preds)
    #print("Length:", length)

    em_mse=mse_cal(emittance_preds,emittance_actuals)
    energy_mse=mse_cal(beam_energy_preds,beam_energy_actuals)
    spread_mse=mse_cal(beam_spread_preds,beam_spread_actuals)
    #print("My calculated emittance mse:",em_mse)
    #print("My calculated beam energy mse:",energy_mse)
    #print("My calculated beam spread mse:",spread_mse)

    # train_em_mse=mse_cal(train_emittance_preds,train_emittance_actuals)
    # train_energy_mse=mse_cal(train_beam_energy_preds,train_beam_energy_actuals)
    # train_spread_mse=mse_cal(train_beam_spread_preds,train_beam_spread_actuals)

    emittance_loss_array=np.append(emittance_loss_array,em_mse)
    energy_loss_array=np.append(energy_loss_array,energy_mse)
    spread_loss_array=np.append(spread_loss_array,spread_mse)



#print("Emittance train loss array:",train_emittance_loss_array)
#print("Energy train loss array:",train_energy_loss_array)
#print("Spread train loss array:",train_spread_loss_array)

print("MSE____________________________")
print("Emittance loss:",np.mean(emittance_loss_array),"+/-",np.std(emittance_loss_array))
print("Beam Energy loss:",np.mean(energy_loss_array),"+/-",np.std(energy_loss_array))
print("Beam Spread loss:",np.mean(spread_loss_array),"+/-",np.std(spread_loss_array))

print("Train Emittance loss:",np.mean(train_emittance_loss_array),"+/-",np.std(train_emittance_loss_array))
print("Train Beam Energy loss:",np.mean(train_energy_loss_array),"+/-",np.std(train_energy_loss_array))
print("Train Beam Spread loss:",np.mean(train_spread_loss_array),"+/-",np.std(train_spread_loss_array))




print("RMSE____________________________")
print("Emittance loss:",np.mean(np.sqrt(emittance_loss_array)),"+/-",np.std(np.sqrt(emittance_loss_array)))
print("Beam Energy loss:",np.mean(np.sqrt(energy_loss_array)),"+/-",np.std(np.sqrt(energy_loss_array)))
print("Beam Spread loss:",np.mean(np.sqrt(spread_loss_array)),"+/-",np.std(np.sqrt(spread_loss_array)))

print("Train Emittance loss:",np.mean(np.sqrt(train_emittance_loss_array)),"+/-",np.std(np.sqrt(train_emittance_loss_array)))
print("Train Beam Energy loss:",np.mean(np.sqrt(train_energy_loss_array)),"+/-",np.std(np.sqrt(train_energy_loss_array)))
print("Train Beam Spread loss:",np.mean(np.sqrt(train_spread_loss_array)),"+/-",np.std(np.sqrt(train_spread_loss_array)))

em_rmse=np.mean(np.sqrt(emittance_loss_array))
energy_rmse=np.mean(np.sqrt(energy_loss_array))
spread_rmse=np.mean(np.sqrt(spread_loss_array))

#plotting
model.eval()
# Extract all test data in one go
all_features = torch.cat([batch[0] for batch in test_dataloader], dim=0).to(device)
all_targets = torch.cat([batch[1] for batch in test_dataloader], dim=0).to(device)

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

length=len(emittance_preds)
print("Length:", length)



#print("Beam spread predictions:",beam_spread_preds)
#print("Actual Beam spread:",beam_spread_actuals)


# Calculate residuals
beam_spread_residuals =  beam_spread_preds-beam_spread_actuals
beam_energy_residuals =  beam_energy_preds-beam_energy_actuals 
emittance_residuals = emittance_preds-emittance_actuals 

em_x=np.linspace(min(min(emittance_actuals), min(emittance_preds)),max(max(emittance_actuals), max(emittance_preds)),100)
em_y_upper = em_x + em_rmse
em_y_lower = em_x - em_rmse

spread_x=np.linspace(min(min(beam_spread_actuals), min(beam_spread_preds)),max(max(beam_spread_actuals), max(beam_spread_preds)),100)
spread_y_upper = spread_x + spread_rmse
spread_y_lower = spread_x - spread_rmse

energy_x=np.linspace(min(min(beam_energy_actuals), min(beam_energy_preds)),max(max(beam_energy_actuals), max(beam_energy_preds)),100)
energy_y_upper = energy_x + energy_rmse
energy_y_lower = energy_x - energy_rmse

# Create figure and GridSpec
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.4)

# Scatter plots
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(beam_spread_actuals, beam_spread_preds, color='tab:blue', alpha=0.6,label="Beam Spread Data")
ax1.plot(beam_spread_actuals, beam_spread_actuals, color='k', linestyle='-',label="y=x")
ax1.fill_between(spread_x,spread_y_lower,spread_y_upper , color="red", alpha=0.2, label=(r"RMSE: "+str(np.round(spread_rmse,3))))
ax1.set_title('Beam Spread: Simulation vs Predicted')
ax1.set_xlabel('Simulation Beam Spread')
ax1.set_ylabel('Predicted Beam Spread')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(beam_energy_actuals, beam_energy_preds, color='tab:green', alpha=0.6,label="Beam Energy Data")
ax2.plot(beam_energy_actuals, beam_energy_actuals, color='k', linestyle='-',label="y=x")
ax2.fill_between(energy_x,energy_y_lower,energy_y_upper , color="red", alpha=0.2, label=(r"RMSE: "+str(np.round(energy_rmse,3))))
ax2.set_title('Beam Energy: Simulation vs Predicted')
ax2.set_xlabel('Simulation Beam Energy')
ax2.set_ylabel('Predicted Beam Energy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(emittance_actuals, emittance_preds, color='tab:purple', alpha=0.6,label="Emittance Data")
ax3.plot(emittance_actuals, emittance_actuals, color='k', linestyle='-',label="y=x")
ax3.fill_between(em_x,em_y_lower,em_y_upper , color="red", alpha=0.2, label=(r"RMSE: "+str(np.round(em_rmse,3))))
ax3.set_title('Emittance: Simulation vs Predicted')
ax3.set_xlabel('Simulation Emittance')
ax3.set_ylabel('Predicted Emittance')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.5)

# Residual plots
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(beam_spread_actuals, beam_spread_residuals/spread_rmse, color='tab:blue', alpha=0.6)
ax4.axhline(0, color='k', linestyle='-')
ax4.axhline(-1,color='r',linestyle='--',linewidth=1)
ax4.axhline(1,color='r',linestyle='--',linewidth=1)
ax4.set_title('Beam Spread Residuals')
ax4.set_xlabel('Actual Beam Spread')
ax4.set_ylim(-3,3)
ax4.set_ylabel('Residual')
ax4.grid(True, linestyle='--', alpha=0.5)

ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(beam_energy_actuals, beam_energy_residuals/energy_rmse, color='tab:green', alpha=0.6)
ax5.axhline(0, color='k', linestyle='-')
ax5.axhline(-1,color='r',linestyle='--',linewidth=1)
ax5.axhline(1,color='r',linestyle='--',linewidth=1)
ax5.set_title('Beam Energy Residuals')
ax5.set_xlabel('Actual Beam Energy')
ax5.set_ylim(-3,3)
ax5.set_ylabel('Residual')
ax5.grid(True, linestyle='--', alpha=0.5)

ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(emittance_actuals, emittance_residuals/em_rmse, color='tab:purple', alpha=0.6)
ax6.axhline(0, color='k', linestyle='-')
ax6.axhline(-1,color='r',linestyle='--',linewidth=1)
ax6.axhline(1,color='r',linestyle='--',linewidth=1)
ax6.set_title('Emittance Residuals')
ax6.set_xlabel('Actual Emittance')
ax6.set_ylim(-3,3)
ax6.set_ylabel('Residual')
ax6.grid(True, linestyle='--', alpha=0.5)
plt.savefig(r'Machine Learning\plots\emittance_energy_spread_2000.png', dpi=300, bbox_inches='tight')
plt.show()
