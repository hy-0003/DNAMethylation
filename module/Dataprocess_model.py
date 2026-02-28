import os
import torch
from util import util_file
import torch.utils.data as Data


class DataProcessor:
    def __init__(self, dataset_name):
        parts = dataset_name.split('_', 1) 
        if len(parts) < 2:
            raise ValueError(f"数据集名称格式错误：{dataset_name}，应为 '修饰类型_物种' 格式(如 4mC_C.equisetifolia)")
        modification_type = parts[0]
        path = os.path.join(modification_type, dataset_name)

        self.train_path = f'data/DNA_MS/tsv/{path}/train.tsv'
        self.test_path = f'data/DNA_MS/tsv/{path}/test.tsv'


class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        # label:
        self.train_label = None
        self.test_label = None
        #
        self.train_dataset = None
        self.test_dataset = None
        # 
        self.train_dataloader = None
        self.test_dataloader = None

        # These fields will be initialized when load_data() is called and original data loaded
        self.class0_idx_full = []
        self.class1_idx_full = []
        self.class0_remaining = []
        self.class1_remaining = []
        self.class0_total = 0
        self.class1_total = 0


    def load_data(self):
        # util_file.load_tsv_format_data 应返回 (data_list, label_list)
        self.train_dataset, self.train_label = util_file.load_tsv_format_data(self.config.path_train_data)
        self.test_dataset, self.test_label = util_file.load_tsv_format_data(self.config.path_test_data)

        self.train_dataloader = self.construct_dataset(self.train_dataset, self.train_label, self.config.cuda,
                                                           self.config.batch_size)
        self.test_dataloader = self.construct_dataset(self.test_dataset, self.test_label, self.config.cuda,
                                                          self.config.batch_size)


    def construct_dataset(self, sequences, labels, cuda, batch_size):
        if cuda:  
            labels = torch.tensor(labels, dtype=torch.long, device='cuda')
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet(sequences, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      shuffle=True)
        print('len(data_loader)', len(data_loader))
        return data_loader

    def get_dataloder(self, name):
        if name == 'train_set':
            return self.train_dataloader
        elif name == 'test_set':
            return self.test_dataloader
        return None


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
