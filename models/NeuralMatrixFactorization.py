import torch
from torch import nn

class NeuralMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, weight_decay=0.01):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.neural_network = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 1),
        )

        self.user_embeddings.weight.data.uniform_(-0.05, 0.05)
        self.item_embeddings.weight.data.uniform_(-0.05, 0.05)

        # Define weight decay for linear layers
        self.weight_decay = weight_decay

    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embeddings(user_ids)
        item_embedded = self.item_embeddings(item_ids)
        x = torch.cat([user_embedded, item_embedded], dim=1)
        x = self.neural_network(x)
        return x.squeeze()

    def get_regularization_loss(self):
        # Calculate L2 regularization loss for linear layers
        reg_loss = 0.0
        for module in self.neural_network.modules():
            if isinstance(module, nn.Linear):
                reg_loss += 0.5 * self.weight_decay * torch.sum(module.weight ** 2)
        return reg_loss


