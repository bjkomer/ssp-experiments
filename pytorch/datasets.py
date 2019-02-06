import numpy as np
import torch
import torch.utils.data as data


class PathDataset(data.Dataset):

    def __init__(self, ssp_inputs, direction_outputs, coord_inputs):

        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.direction_outputs = direction_outputs.astype(np.float32)
        self.coord_inputs = coord_inputs.astype(np.float32)

    def __getitem__(self, index):
        return self.ssp_inputs[index], self.direction_outputs[index], self.coord_inputs[index]

    def __len__(self):
        return self.ssp_inputs.shape[0]


class GenericDataset(data.Dataset):

    def __init__(self, inputs, outputs):

        self.inputs = inputs.astype(np.float32)
        self.outputs = outputs.astype(np.float32)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return self.inputs.shape[0]
