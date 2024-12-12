import torch
from utils.plot import plot_roc

def eval_model(X_norm,A_norm,A_test,model, link_predictor, device = "cpu",
              predict_indx = 0, loss_all = True, weighted = True, color =  "red"):
    X_eval = X_norm.to(device)
    A_eval = A_norm.to(device)
    A_true = A_test[predict_indx].to(device)
    # Step 2: Calculate link probabilities for each possible node pair
    num_nodes = A_true.shape[-1]
    A_pred = torch.zeros(num_nodes, num_nodes).float()

    with torch.no_grad():
        model.eval()
        link_predictor.eval()
        H_final = model(A_eval, X_eval)
        for u in range(num_nodes):
            for v in range(num_nodes):
                # Get node embeddings
                h_u, h_v = H_final[u], H_final[v]
                # Predict link score/prob
                link_score = link_predictor(h_u, h_v).item()
                A_pred[u, v] = link_score   # Store in adjacency matrix

    if weighted:
        if loss_all:
            # Calculate the loss for all element
            metric = ((A_true - A_pred)**2).mean()
        else:
            # Calculate the loss only for positive elements
            true_edges = torch.nonzero(A_pred > 0, as_tuple=True)
            metric = ((A_true[true_edges[0],true_edges[1]] - A_pred[true_edges[0],true_edges[1]])**2).mean()
        metric = metric.item()
    else:
        if not loss_all:
            print("loss_all does not work with weighted = False" )
        predictions = A_pred.flatten()
        labels = A_true.flatten()
        metric = plot_roc(predictions=predictions,labels=labels, average_precision=True, color=color)
        
    return A_pred, metric
    
def eval_baseline(A_train,A_test,predict_indx = 0, loss_all = True, weighted = True,
                 color = "red"):
    A_pred = A_train[-1]
    A_true = A_test[predict_indx]
    if weighted:
        if loss_all:
            # Calculate the loss for all element
            metric = ((A_true - A_pred)**2).mean()
        else:
            # Calculate the loss only for positive elements
            true_edges = torch.nonzero(A_pred > 0, as_tuple=True)
            metric = ((A_true[true_edges[0],true_edges[1]] - A_pred[true_edges[0],true_edges[1]])**2).mean()
        # MSE
        metric = metric.item()
    else:
        if not loss_all:
            print("loss_all does  not work with weighted = False" )
        predictions = A_pred.flatten()
        labels = A_true.flatten()
        metric = plot_roc(predictions=predictions,labels=labels, average_precision=True, color=color)
        
    return A_pred, metric