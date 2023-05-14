import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import random_seed
from datasets import UserItemRatingDataset
from models import NeuralMatrixFactorization

# Define the training loop
def train_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    for user_ids, item_ids, ratings in progress_bar:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions.squeeze(), ratings.type(torch.float32))
        loss.backward()
        optimizer.step()

        # Update the total loss
        total_loss += loss.item()

    # Compute the average loss
    average_loss = total_loss / len(dataloader)

    return average_loss

# Define the training loop with regularization loss
def train_loop_reg(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_reg_loss = 0  
    progress_bar = tqdm(dataloader, desc='Training')
    for user_ids, item_ids, ratings in progress_bar:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions.squeeze(), ratings.type(torch.float32))

        reg_loss = model.get_regularization_loss()  # Calculate regularization loss
        total_reg_loss += reg_loss.item()  # Accumulate regularization loss

        # Add regularization loss to the main loss
        loss += reg_loss

        loss.backward()
        optimizer.step()

        # Update the total loss
        total_loss += loss.item()

    # Compute the average losses
    average_loss = total_loss / len(dataloader)
    average_reg_loss = total_reg_loss / len(dataloader)

    return average_loss



def validate_loop(model, dataloader, criterion, device):
    model.eval()
    average_loss = 0
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions.squeeze(), ratings.type(torch.float32))
            losses.append(loss.item())
            y_pred.append(predictions.cpu().numpy())
            y_true.append(ratings.cpu().numpy())

    # Calculate metrics
    all_predictions = np.concatenate(y_pred)
    all_targets = np.concatenate(y_true)
    all_predictions = np.around(all_predictions, decimals=0)
    
    # Exclude class 0
    mask = all_targets != 0
    targets_without_zero = all_targets[mask]
    predictions_without_zero = all_predictions[mask]
    
    precision = precision_score(targets_without_zero, predictions_without_zero, average='micro')
    recall = recall_score(targets_without_zero, predictions_without_zero, average='micro')

    mae = mean_absolute_error(all_targets, all_predictions)
    average_loss = sum(losses) / len(losses)

    return average_loss, precision, recall, mae


def train(
        csv_file:str, user_id_col:str, item_id_col:str, rating_col:str, 
        model_file:str="NeuralMF.pt", train_split_ratio=0.8, 
        epochs=40, lr=0.0009, batch_size=512, embedding_dim=10
    ):
    # Load the dataset
    users_items_ratings = pd.read_csv(csv_file)

    dataset = UserItemRatingDataset(users_items_ratings, user_id_col=user_id_col, item_id_col=item_id_col, rating_col=rating_col)
        
    train_size = int(train_split_ratio* len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define the PyTorch dataloader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=True)#int(batch_size/2), shuffle=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the collaborative filtering model
    model = NeuralMatrixFactorization(dataset.num_users, dataset.num_items, embedding_dim=embedding_dim)
    #model = torch.load(model_file).to(device) # to load existing model
    model = model.to(device)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    random_seed(0)
    train_losses, val_losses = list(), list()
    precisions, recalls, maes = list(), list(), list()

    for epoch in range(epochs):
        #random_seed(epoch)
        
        train_loss = train_loop_reg(model, train_loader, criterion, optimizer, device)
        val_loss, precision, recall, mae = validate_loop(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        recalls.append(precision)
        precisions.append(recall)
        maes.append(mae)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, precision: {precision:.3f}, recall: {recall:.3f}, mae: {mae:.3f}')

    print("Training Complete")
    
    torch.save(model, model_file)
    
    # Plotting Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Training_Validation_Loss.png")

    # Plotting Precision
    plt.figure(figsize=(10, 6))
    plt.plot(precisions, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()
    plt.savefig("Precision.png")

    # plotting Recall
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()
    plt.savefig("Recall.png")

    # Plotting NDCG
    plt.figure(figsize=(10, 6))
    plt.plot(maes, label='mae')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.savefig("MAE.png")

#train(csv_file="data/sample_interactions.csv", user_id_col="user_id", item_id_col="recipe_id", rating_col="rating", model_file="NeuralMF.pt")
