import pandas as pd
import numpy as np
import torch

from utils import log_transform, unweight_adj

def load_features(PATH_node_features, years, feature_names = ["GDP","GINI","HDI","PTS","U5M"],
                 remove_nan = True):

    ## Reading feature files
    features = []
    if "GDP" in feature_names:
        GDP_PATH = "GDP_per_Capita.csv"
        GDP = pd.read_csv(PATH_node_features + GDP_PATH)
        features.append(GDP)
    if "GINI" in feature_names:
        GINI_PATH = "GINI_Index.csv"
        GINI = pd.read_csv(PATH_node_features + GINI_PATH)
        features.append(GINI)
    if "HDI" in feature_names:
        HDI_PATH = "Human_Development_Index.csv"
        HDI = pd.read_csv(PATH_node_features + HDI_PATH)
        features.append(HDI)
    if "PTS" in feature_names:
        PTS_PATH = "Political_Terror_Scale.csv"
        PTS = pd.read_csv(PATH_node_features + PTS_PATH)
        features.append(PTS)
    if "U5M" in feature_names:
        U5M_PATH = "Under_5_mortality.csv"
        U5M = pd.read_csv(PATH_node_features + U5M_PATH)
        features.append(U5M)
    X_all = []
    for year in years:
        x = []
        for feature in features:
            # Extract features from each year
            x.append((feature[str(year)].to_numpy()).reshape(-1,1))
        # Concat features for each node together
        X_all.append(np.concatenate(x,axis = 1))
    X_all = np.array(X_all) # 29 (year) x 232 (countries) x 5 (features)
    if remove_nan:
        countries_codes = pd.read_csv(PATH_node_features + "Country_codes_names.csv")
        countries = list(countries_codes["Name"])
        nan_idx = np.unique(np.where(np.isnan(X_all))[1])
        nan_countries = np.array(countries)[nan_idx]
        print("Removed countries:",len(nan_countries))
        print("Remaining countries:", 232 - len(nan_countries))
        non_nan_idx = np.ones(len(countries),dtype =bool)
        non_nan_idx[nan_idx] = False
        X_np = X_all[:,non_nan_idx,:]
        
        return X_np,non_nan_idx
    else:
        return X_all
    
def load_edges(PATH_EDGES, log_transfrom_data = True, remove_self_loops = False,
              non_nan_idx = None, mode = "refugee", years_range = np.arange(1992,2020 + 1)):
    """
    If non_nan_idx is None then no countries is removed
    """
    assert mode in ["refugee","migration"]
    ### Making the Adjecency Matrix
    ## Dynamic Path to edge files
    if mode == "refugee":
        PATH_ref = lambda year: f"Refugee_Stock_{year}.csv"
        years = years_range
    elif mode == "migration":
        PATH_ref = lambda year: f"Migration_Stock_{year}.csv"
        years = np.array([1995,2000,2005,2010,2015,2020])     
    A = []
    for year in years:
        # Read csv file and NaN -> 0
        adj = pd.read_csv(PATH_EDGES + PATH_ref(year)).fillna(0)
        # Remove id column and make to np array
        adj = adj.drop(columns = [str(year)]).to_numpy()
        if log_transfrom_data:
            adj = log_transform(adj)
        if remove_self_loops:
            np.fill_diagonal(adj,0)
        A.append(adj)

    A = np.array(A)
    if non_nan_idx is not None:
        # Remove nodes with NaN features
        A = A[:,non_nan_idx]
        A = A[:,:,non_nan_idx]
    return A

def prepare_language(PATH, non_nan_idx = None):
    LAN_PATH = "Language_Used.csv"
    LAN = pd.read_csv(PATH + LAN_PATH)
    LAN_np = LAN.to_numpy()
    LAN_np[:,0] = LAN_np[:,0] - 1
    for i in range(LAN_np.shape[0]):
        row = LAN_np[i]
        if np.isnan(row).any():
            idx,first,second,third = row
            if np.isnan(second):
                second = first
                third = first
            else:
                third = second
            LAN_np[i] = [idx,first,second,third]

    # Remove indx:
    LAN_np = LAN_np[:,1:]
    if non_nan_idx is not None:
        LAN_np = LAN_np[non_nan_idx]
    return LAN_np

def normalize_adjacency(A):
    #print(A.shape)
    """
    Normalize adjacency matrix using the symmetric normalization method.
    :param A: Raw adjacency matrix (time, num_vertices, num_vertices)
    :return: Normalized adjacency matrix Ab (time, num_vertices, num_vertices)
    """
    time_steps, num_vertices, _ = A.shape
    I = torch.eye(num_vertices, device=A.device).unsqueeze(0).repeat(time_steps, 1, 1)  # Identity matrix for each time
    A_hat = A + I

    D_hat = torch.diag_embed(A_hat.sum(dim=-1).pow(-0.5))  # D^(-1/2)

    A_normalized = torch.bmm(D_hat, torch.bmm(A_hat, D_hat))
    return A_normalized

def normalize_features(X):
    # Assume X has shape [T, N, F]
    T, N, F = X.shape

    # Initialize a new tensor to store normalized features
    X_normalized = torch.empty_like(X)

    # Normalize features for each time step
    for t in range(T):
        # Extract features for the t-th time step (shape: [N, F])
        X_t = X[t]

        # Calculate mean and std for each feature across all nodes at this time step
        mean = X_t.mean(dim=0, keepdim=True)  # Shape: [1, F]
        std = X_t.std(dim=0, keepdim=True)    # Shape: [1, F]

        # Apply normalization (zero mean, unit variance)
        X_normalized[t] = (X_t - mean) / (std + 1e-5)
    return X_normalized

def prepare_data(A,X, tt_idx = 23, embedding_dim = 4, LAN_np = None, weighted = True):
    A = A.float()
    X = X.float()
    
    if not weighted:
        A = unweight_adj(A)

    A_train, X_train = (A[:tt_idx],X[:tt_idx])
    A_test, X_test = (A[tt_idx:],X[tt_idx:])

    # Normalize adjacency matrix
    A_norm = normalize_adjacency(A_train)
    # Normalize features
    X_norm = normalize_features(X_train)
    if LAN_np is not None:
        # Tranform to tensor + use_embedding dim
        LAN_t = torch.from_numpy(LAN_np).long() - 1
        embedding = torch.nn.Embedding(num_embeddings=((LAN_t).max().long()+1), embedding_dim=embedding_dim)
        LAN_EMB = embedding(LAN_t).reshape(LAN_t.shape[0],-1)
        LAN_EMB_expanded = LAN_EMB.unsqueeze(0).expand(tt_idx, -1, -1)  # Now has shape [29, 130, 12]
        X_norm = torch.cat((X_norm, LAN_EMB_expanded), dim=-1).detach()
    return A_norm,X_norm,A_train,A_test,X_test
