import time
import torch

def get_edge_indices(A_train):
    """
    Input: 
    A_train - torch tensor (timesteps x nodes x nodes)
    Return:
    positive_edges - torch tensor (pos_indices x 2)
    negative_edges - torch tensor (neg_indices x 2)
    """
    # Precompute the list of positive and negative edges for each timestep
    positive_edges = []
    negative_edges = []

    for t in range(A_train.shape[0] - 1):  # T-1, as we predict A_{t+1} from t
        A_target = A_train[t+1]  # Target adjacency matrix at time t+1 (original A)

        # Positive edges: Non-zero entries in the adjacency matrix
        pos_edges_t = torch.nonzero(A_target > 0)
        positive_edges.append(pos_edges_t)

        # Negative edges: Zero entries in the adjacency matrix
        neg_edges_t = torch.nonzero(A_target == 0)  # Find zero entries (no edges)
        negative_edges.append(neg_edges_t)
    return positive_edges, negative_edges

def sample_edges(edges, max_samples):
    """
    Return sampled indices as a (n x 2) tensor
    """
    # Ensure that max samples is not higher than len(edges)
    n = len(edges)
    num_positive_samples = min(n, max_samples)
    sample_indices =  torch.randperm(n)[:num_positive_samples]
    sampled_edges = edges[sample_indices]
    return sampled_edges

def edge_loss(H_final,A_target,edges,link_predictor, criterion, calc_reverse):
    """
    H_final: Final hidden embedding of the nodes
    edges : edge indices (max_samples x 2) tensor
    link_predictor: MLP to calculate the pairwise score/prob 
    A_target: n x n true target adj matrix
    criterion: loss function
    calc_reverse: include reverse order of nodes
    """
    device = A_target.device
    
    pos_preds = []
    neg_preds = []
    pos_targets = []
    for (u,v) in edges:
        h_u, h_v = H_final[u], H_final[v]
        pred = link_predictor(h_u, h_v)
        target = A_target[u, v]
        if target > 0:
            pos_preds.append(pred)
            pos_targets.append(target)
        else:
            neg_preds.append(pred)
        if calc_reverse:
            reverse_pred = link_predictor(h_v, h_u)
            reverse_target = A_target[v,u] 
            if reverse_target > 0:
                pos_preds.append(reverse_pred)
                pos_targets.append(reverse_target)
            else:
                neg_preds.append(reverse_pred)
    #if pos_preds is not empty
    if len(pos_preds) > 0:
        pos_preds = torch.stack(pos_preds).squeeze()
        if len(pos_preds.shape) == 0:
               pos_preds = pos_preds.unsqueeze(0)
        pos_targets = torch.tensor(pos_targets, dtype=torch.float32, device=device)
        pos_loss = criterion(pos_preds, pos_targets)
        
    else:
        pos_loss = 0.0
    #if neg_preds is not empty
    if len(neg_preds) > 0:
        neg_preds = torch.stack(neg_preds).squeeze()
        if len(neg_preds.shape) == 0:
               neg_preds = neg_preds.unsqueeze(0)
        neg_targets = torch.zeros_like(neg_preds)
        neg_loss = criterion(neg_preds, neg_targets)
    else:
        neg_loss = 0.0
    return pos_loss, neg_loss

def train(X_norm,A_norm,A_train,model,link_predictor,criterion,optimizer,device,
          num_epochs = 200,edge_subset= 200,calc_reverse = False, loss_weights = [1.,1.],
          save_at = 50, weighted = True):
    """
    Inputs:
    model - Model to train
    X_norm - Normalized node features (time x nodes x feature)
    A_norm - Normalized adj matrix (D^{-1/2}(A + I)D^{-1/2}) (time x nodes x nodes)
    A_train - Orignal (log) adj matrices (time x nodes x nodes)
    criterion - loss function (MSE or BCE)
    optimizer - optimizer (ADAM)
    device - cpu or cuda
    num_epochs - number of epochs to train
    edge_subset - number of edges to sample during training
    calc_reverse - calculate the revser order of nodes (for directed graphs)
    negative_weight - down or up weight the influence of the negative edges
    save_at - save model and print loss at save_at epoch
    """

    losses = []
    # Calculate positive and negative edges
    positive_edges, negative_edges = get_edge_indices(A_train)
    # Start training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time() 
        model.train()  # Set model to training mode
        link_predictor.train()

        # Loop over each time series step
        for t in range(X_norm.shape[0] - 1):  # T-1, as we predict A_{t+1} from t
            # Input at time t and target adjacency at time t+1
            X_t = X_norm[:t+1]          # Node features up to time t
            A_t = A_norm[:t+1]          # Adjacency matrix up to time t (for GCN layer)
            A_target = A_train[t+1]     # Target adjacency matrix at time t+1 (original A)

            # Move data to the correct device (e.g., GPU if available)
            X_t, A_t, A_target = X_t.to(device), A_t.to(device), A_target.to(device)

            # Forward pass: Generate node embeddings
            H_final = model(A_t, X_t)  # Use model to get node embeddings

            # Sample a subset of positive and negative edges
            sampled_pos_edges = sample_edges(positive_edges[t], max_samples= edge_subset)
            sampled_neg_edges = sample_edges(negative_edges[t], max_samples= edge_subset)

            # --- Pairwise Score Prediction (Positive and Negative Edges) ---

            pos_loss1, neg_loss1 = edge_loss(H_final,A_target,sampled_pos_edges,link_predictor,criterion,
                                          calc_reverse)

            pos_loss2, neg_loss2 = edge_loss(H_final,A_target,sampled_neg_edges,link_predictor,criterion,
                                          calc_reverse)


            # Combine positive and negative loss
            positive_weight, negative_weight = loss_weights
            combined_loss = positive_weight * (pos_loss1 + pos_loss2) + negative_weight * (neg_loss1 + neg_loss2)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            combined_loss.backward()  # Compute gradients
            optimizer.step()       # Update model parameters

            # Track epoch loss
            epoch_loss += combined_loss.item()
        epoch_time = time.time() - start_time
        # Save losses
        losses.append(epoch_loss)
        # Save model & print average loss
        if epoch % save_at == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
            torch.save({
                'model_state_dict': model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': losses,
                }, 'checkpoint.pth')
        
    # Saving the final model
    mode = "weighted" if weighted else "unweighted"
    gcn = "GCN_MA" if model.attention else "EvolveGCN"
    torch.save({'model_state_dict': model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': losses,
                }, f'{gcn}_{mode}.pth')
    print("Training complete.")
    
    return losses