import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset


class DataNormer(object):
    """
        data normalization at last dimension
    """

    def __init__(self, data, method="min-max", axis=None):
        """
            data normalization at last dimension
            :param data: data to be normalized
            :param method: normalization method
            :param axis: axis to be normalized
        """
        if isinstance(data, str):
            if os.path.isfile(data):
                try:
                    self.load(data)
                except:
                    raise ValueError("the savefile format is not supported!")
            else:
                raise ValueError("the file does not exist!")
        elif type(data) is np.ndarray:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data, axis=axis)
                self.min = np.min(data, axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data, axis=axis)
                self.std = np.std(data, axis=axis)
        elif type(data) is torch.Tensor:
            if axis is None:
                axis = tuple(range(len(data.shape) - 1))
            self.method = method
            if method == "min-max":
                self.max = np.max(data.numpy(), axis=axis)
                self.min = np.min(data.numpy(), axis=axis)

            elif method == "mean-std":
                self.mean = np.mean(data.numpy(), axis=axis)
                self.std = np.std(data.numpy(), axis=axis)
        else:
            raise NotImplementedError("the data type is not supported!")


    def norm(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std + 1e-10, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + 1e-10) + torch.tensor(self.min,
                                                                                                     device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std + 1e-10, device=x.device)) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + 1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x
    def save(self, save_path):
        """
            save the parameters to the file
            :param save_path: file path to save
        """
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path):
        """
            load the parameters from the file
            :param save_path: file path to load
        """
        import pickle
        isExist = os.path.exists(save_path)
        if isExist:
            try:
                with open(save_path, 'rb') as f:
                    load = pickle.load(f)
                self.method = load.method
                if load.method == "mean-std":
                    self.std = load.std
                    self.mean = load.mean
                elif load.method == "min-max":
                    self.min = load.min
                    self.max = load.max
            except:
                raise ValueError("the savefile format is not supported!")
        else:
            raise ValueError("The pkl file is not exist, CHECK PLEASE!")


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class GSLoader(object):
    def __init__(self, datapath):
        dataloader = MatReader(datapath)

        self.field = dataloader.read_field('psitmp').unsqueeze(-1)[:, ::5, ::5, :]                 # (10000, 161, 101, 1)
        self.coords = np.concatenate((dataloader.read_field('RR').unsqueeze(-1)[:, ::5, ::5, :] , 
                                      dataloader.read_field('ZZ').unsqueeze(-1)[:, ::5, ::5, :] ), axis=-1)  # (10000, 161, 101, 2)
        self.coords_Rin = dataloader.read_field('Rtmp').permute(0, 2, 1)[:, ::50, :]          # (10000, 361, 1)
        self.coords_Zin = dataloader.read_field('Ztmp').permute(0, 2, 1)[:, ::50, :]        
        self.coords_inter = torch.cat((self.coords_Rin, self.coords_Zin), dim=-1)  # (10000, 361, 2)

        # self.coords_s = self.coords.shape[0]
        # self.fields_s = self.field.shape[0]

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        Xs = self.coords[start:start + n_sample]
        ys = self.field[start:start + n_sample]
        Xs_inter = self.coords_inter[start:start + n_sample]

        # 归一化
        Xs_normalizer = DataNormer(Xs, method='mean-std')
        Xs = Xs_normalizer.norm(Xs)

        ys_normalizer = DataNormer(ys, method='mean-std')
        ys = ys_normalizer.norm(ys)

        Xs_inter_normalizer = DataNormer(Xs_inter, method='mean-std')
        Xs_inter = Xs_inter_normalizer.norm(Xs_inter)

        dataset = torch.utils.data.TensorDataset(torch.tensor(Xs), torch.tensor(ys), torch.tensor(Xs_inter))
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader