import torch
from pandas import DataFrame
from torch.utils.data import Dataset

class UserItemRatingDataset(Dataset):
    def __init__(self, dataset, user_id_col, item_id_col, rating_col):
        self.dataset = dataset.copy()
        
        # Extract unique user IDs and item IDs
        users = dataset[user_id_col].unique()
        items = dataset[item_id_col].unique()
        
        # Assign continuous IDs to users and items
        self.userid2idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.itemid2idx = {item_id: idx for idx, item_id in enumerate(items)}
        
        # Map continuous IDs back to original IDs
        self.idx2userid = {idx: user_id for user_id, idx in self.userid2idx.items()}
        self.idx2itemid = {idx: item_id for item_id, idx in self.itemid2idx.items()}

        self.num_users = len(users)
        self.num_items = len(items)
        
        # Update dataset with continuous IDs
        self.dataset["item_index"] = dataset[item_id_col].apply(lambda x: self.itemid2idx[x])
        self.dataset["user_index"] = dataset[user_id_col].apply(lambda x: self.userid2idx[x])
        
        # Prepare input features and labels as tensors
        self.user_ids = torch.tensor(self.dataset["user_index"].values)
        self.item_ids = torch.tensor(self.dataset["item_index"].values)
        self.ratings = torch.tensor(self.dataset[rating_col].values, dtype=torch.float32)
        
    def __getitem__(self, index):
        return (self.user_ids[index], self.item_ids[index], self.ratings[index])

    def __len__(self):
        return len(self.dataset)
