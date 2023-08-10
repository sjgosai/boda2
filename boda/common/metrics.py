import torch

def jaccard_index(x, y, max_batch_size=1024, dtype=torch.float, device='cpu'):
    """
    Compute the Jaccard index (intersection over union) between two sets of vectors.

    Args:
        x (torch.Tensor): First set of vectors, shape (N, D).
        y (torch.Tensor): Second set of vectors, shape (M, D).
        max_batch_size (int): Maximum batch size for DataLoader (default: 1024).
        dtype (torch.dtype): Data type for calculations (default: torch.float).
        device (str): Device for calculations (default: 'cpu').

    Returns:
        torch.Tensor: Matrix of Jaccard indices between x and y, shape (N, M).
    """
    assert len(x.shape) == 2, f"expected x as 2D tensor, instead got shape {x.shape}"
    assert len(y.shape) == 2, f"expected y as 2D tensor, instead got shape {y.shape}"
    matrix = []
    x_loader = torch.utils.data.DataLoader( x, batch_size=max_batch_size )
    y_loader = torch.utils.data.DataLoader( y, batch_size=max_batch_size )
    for x_batch in x_loader:
        row = []
        x_batch = x_batch.unsqueeze(1)
        for y_batch in y_loader:
            y_batch = y_batch.unsqueeze(0)
            inter = torch.logical_and(x_batch,y_batch).sum(-1)
            union = torch.logical_or(x_batch,y_batch).sum(-1)
            ji = inter / union
            row.append(ji.type(dtype).to(device))
        matrix.append( torch.cat(row,dim=-1) )
    return torch.cat(matrix,dim=0)

def euclidean_distance(x, y, max_batch_size=1024, dtype=torch.float, device='cpu'):
    """
    Compute the Euclidean distance between two sets of vectors.

    Args:
        x (torch.Tensor): First set of vectors, shape (N, D).
        y (torch.Tensor): Second set of vectors, shape (M, D).
        max_batch_size (int): Maximum batch size for DataLoader (default: 1024).
        dtype (torch.dtype): Data type for calculations (default: torch.float).
        device (str): Device for calculations (default: 'cpu').

    Returns:
        torch.Tensor: Matrix of Euclidean distances between x and y, shape (N, M).
    """
    assert len(x.shape) == 2, f"expected x as 2D tensor, instead got shape {x.shape}"
    assert len(y.shape) == 2, f"expected y as 2D tensor, instead got shape {y.shape}"
    matrix = []
    x_loader = torch.utils.data.DataLoader( x, batch_size=max_batch_size )
    y_loader = torch.utils.data.DataLoader( y, batch_size=max_batch_size )
    for x_batch in x_loader:
        row = []
        x_batch = x_batch.unsqueeze(1)
        for y_batch in y_loader:
            y_batch = y_batch.unsqueeze(0)
            dist  = (x_batch - y_batch).pow(2).sum(-1).pow(0.5)
            row.append(dist.type(dtype).to(device))
        matrix.append( torch.cat(row,dim=-1) )
    return torch.cat(matrix,dim=0)