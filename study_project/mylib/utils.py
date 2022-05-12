import numpy as np


def data_set_to_dict(dataX, dataY):
    data_dict = {}
    for i in range(len(dataX)):
        ans = int([np.where(dataY[i] == 1.0)][0][0][0])
        if ans in data_dict:
            data_dict[ans].append(dataX[i])
        else:
            data_dict[ans] = [dataX[i]]
    for k, v in data_dict.items():
        data_dict[k] = np.array(v)
    return data_dict


def dimensional_reduction_umap(data):
    pass
