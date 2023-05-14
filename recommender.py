import torch


def recommend_to_user(model, user_index, top_n):
    device = next(model.parameters()).device  # Get the device of the model
    user_embedding = model.user_embeddings(torch.LongTensor([user_index]).to(device))
    item_embeddings = model.item_embeddings.weight.to(device)
    dot_product = torch.matmul(user_embedding, item_embeddings.transpose(0, 1))
    predicted_ratings = dot_product.squeeze()

    _, indices = torch.topk(predicted_ratings, top_n)

    return indices

def recommend_items(model, preferred_items, top_n):
    device = next(model.parameters()).device  # Get the device of the model
    item_embeddings = model.item_embeddings.weight.to(device)
    preferred_indices = torch.LongTensor(preferred_items).to(device)
    preferred_embeddings = item_embeddings[preferred_indices]
    
    user_embedding = torch.mean(preferred_embeddings, dim=0, keepdim=True)
    
    dot_product = torch.matmul(user_embedding, item_embeddings.transpose(0, 1))
    predicted_ratings = dot_product.squeeze()
    
    _, indices = torch.topk(predicted_ratings, top_n)

    return indices

def recommend(model, user_index, preferred_items, top_n):
    device = next(model.parameters()).device  # Get the device of the model
    user_ids = torch.LongTensor([user_index]).to(device)
    item_ids = torch.LongTensor(preferred_items).to(device)

    with torch.no_grad():
        user_embedding = model.user_embeddings(user_ids)
        item_embeddings = model.item_embeddings(item_ids)

        user_embedding += torch.mean(item_embeddings, dim=0, keepdim=True)

        x = torch.cat([user_embedding.repeat(item_embeddings.size(0), 1), item_embeddings], dim=1)
        predicted_ratings = model.neural_network(x)

    _, indices = torch.topk(predicted_ratings.squeeze(), top_n)

    return indices
