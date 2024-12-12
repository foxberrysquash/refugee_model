import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear transformations for queries, keys, and values
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        
        # Final linear layer to combine attention head outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, H):
        assert len(H.shape) == 2, "Input tensor H must have shape (N, D) without batch dimension."
        # Shape: (N, D) -> (N, num_heads, head_dim)
        Q = self.query(H).view(H.size(0), self.num_heads, self.head_dim)
        K = self.key(H).view(H.size(0), self.num_heads, self.head_dim)
        V = self.value(H).view(H.size(0), self.num_heads, self.head_dim)
        
        # Compute scaled dot-product attention
        scores = torch.einsum('nqd,nkd->nqk', Q, K) / (self.head_dim ** 0.5)  # Shape: (N, num_heads, num_heads)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Aggregate values
        attended_values = torch.einsum('nqk,nvd->nqd', attention_weights, V)
        
        # Concatenate heads and apply final linear layer
        attended_values = attended_values.reshape(H.size(0), -1)  # Shape: (N, embed_dim)
        return self.out_proj(attended_values)


class WeightUpdateGRU(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(WeightUpdateGRU, self).__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim * output_dim)  # GRU that outputs flattened weight matrix
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    def forward(self, H, W_prev):
        """
        :param H: Node embedding matrix (num_vertices, hidden_dim)
        :param W_prev: Previous weight matrix (output_dim, output_dim)
        :return: Updated weight matrix W_t
        """
        #num_vertices, hidden_dim = H.shape
        num_vertices = H.shape[0]
        #output_dim = W_prev.size(-1)
        # Aggregate H along vertices to form input for GRU
        H_mean = H.mean(dim=0)  # Shape: (hidden_dim,)
        # Flatten W_prev to use as hidden state in GRU
        W_prev_flat = W_prev.view(-1)  # Shape: (output_dim * output_dim,)
        # Update weights with GRU, treating W_prev as hidden state
        W_next_flat = self.gru(H_mean, W_prev_flat)  # Shape: (output_dim * output_dim,)
        # Reshape back to (output_dim, output_dim)
        #W_next = W_next_flat.view(output_dim, output_dim)
        W_next = W_next_flat.view(self.hidden_dim,self.output_dim)
        return W_next



class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, A, H, W):
        """
        Perform graph convolution with dynamically generated weights.
        :param A: Normalized adjacency matrix (num_vertices, num_vertices)
        :param H: Node embeddings (num_vertices, hidden_dim)
        :param W: Weight matrix from GRU (hidden_dim, hidden_dim)
        :return: Updated node embeddings (num_vertices, hidden_dim)
        """
        output = torch.mm(A, torch.mm(H, W))  # GCN operation H(l+1) = (Â H(l) W) + ReLu added in main model body (so it skips the last layer)
        return output

class EGCU_H(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, attention = False, num_heads = 4):
        super(EGCU_H, self).__init__()
        self.num_gcn_layers = num_gcn_layers

        # Initialize GCN layers and GRU for weight update
        output_gcn_dim = hidden_dim if attention else output_dim
        self.dim_list = list(zip(([input_dim] + [hidden_dim]*(num_gcn_layers - 1)),
                                 ([hidden_dim]*(num_gcn_layers - 1) +  [output_gcn_dim])))
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(num_gcn_layers)])
        self.weight_update_grus = nn.ModuleList([WeightUpdateGRU(in_dim, out_dim) for (in_dim,out_dim) in self.dim_list])

        # Initial weight matrix W_0 (one per layer)
        self.W_0 = nn.ParameterList()  # Create a ModuleList to store W_0 for each layer
        for in_dim, out_dim in self.dim_list:
            W = nn.Parameter(torch.randn(in_dim, out_dim))  # Weight matrix for each layer
            nn.init.xavier_uniform_(W)  # Xavier initialization
            self.W_0.append(W)  # Add to the list
        # Attention for GCN_MA
        self.attention = MultiHeadAttention(input_dim=hidden_dim,embed_dim=output_dim,num_heads=num_heads) if attention else attention
    def forward(self, A_log, X):
        # Initialize W_t as W_0 for each layer
        W_t = [W.clone() for W in self.W_0]# Clone W_0 to avoid modifying in place
        H = X  # Start with initial node features

        # Time-wise processing
        for t in range(X.shape[0]):  # Assuming time is the first dimension in X
            A_t = A_log[t]  # Adjacency matrix at time t
            H_t = H[t]      # Node features at time t

            # Temporary list to store updated weights for each layer
            updated_W_t = []

            # Process each GCN layer with GRU weight update
            for l in range(self.num_gcn_layers):
                # Update weights using GRU
                W_new = self.weight_update_grus[l](H_t, W_t[l])
                updated_W_t.append(W_new)


                # Apply GCN layer with updated weights
                H_t = self.gcn_layers[l](A_t, H_t, W_new)
               
                # Apply ReLU activation except in the final layer
                if l != self.num_gcn_layers - 1:
                    H_t = F.relu(H_t)

            # Replace W_t with the updated weights (recreate tensor to avoid in-place operations)
            W_t = [W.clone() for W in updated_W_t]

        if self.attention:
            H_t = self.attention(H_t)
        return H_t   # Final hidden embedding