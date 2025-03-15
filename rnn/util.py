import numpy as np

def multivariate_data(dataset, target, start_idx, end_idx, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    start_idx = start_idx + history_size
    if end_idx is None:
        end_idx = len(dataset) - target_size + 1
    for i in range(start_idx, end_idx):
        idxs = range(i - history_size, i, step)
        data.append(dataset[idxs])
        if single_step:
            labels.append(target[i + target_size - 1])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


def nse(observed, predicted):
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE).
    """
    return 1 - (np.sum((observed - predicted)**2) / np.sum((observed - np.mean(observed))**2))