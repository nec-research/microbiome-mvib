"""
A Dataset class for the microbiome data.
"""
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import os
import pandas as pd
import torch
import numpy as np


label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity / Obesity-joint
        'leaness': 0, 'obesity': 1,
        # Early-Colorectal-EMBL (early = 0, I, II; late = III, IV)
        'crc_0': 0, 'crc_I': 0, 'crc_II': 0, 'crc_III': 1, 'crc_IV': 1,
        # Colorectal-EMBL (healthy = n)
        'CRC_0': 1, 'CRC_I': 1, 'CRC_II': 1, 'CRC_III': 1, 'CRC_IV': 1,
        # Hypertension
        'Health': 0, 'Hypertension': 1,
        # Custom
        'sick': 1,
        # YachidaS
        'healthy': 0, 'CRC-0': 1, 'CRC-I': 1, 'CRC-II': 1, 'CRC-III': 1, 'CRC-IV': 1, 'adenoma': 1,
        'carcinoma_surgery_history': 1
    }


datasets = (
    'Cirrhosis',
    'Colorectal',
    'IBD',
    'Obesity',
    'T2D',
    'WT2D',
    'Obesity-joint',
    'Colorectal-EMBL',
    'Early-Colorectal-EMBL',
    'Hypertension',
    'Delta-Obesity',
    'Delta-Colorectal',
    'Colorectal-YachidaS'
)


datasets_tl = (
    'Cirrhosis',
    'Colorectal',
    'IBD',
    'Obesity',
    'T2D',
    'WT2D',
    'Hypertension',
    'Delta-Obesity',
    'Delta-Colorectal'
)


class MicrobiomeDataset(Dataset):
    """
    Create dataset for a specific disease.
    """
    def __init__(self, data_dir, disease, device, scale=True, data='deafault', diseases=datasets):
        """
        :param data_dir: the path to the raw data
        :param disease: the specific dataset
        :param device: on which device the data should be loaded
        :param scale: use StandardScaler on abundance
        :param data: which dataset collection to use
        :param diseases: the set of diseases
        """
        super(MicrobiomeDataset, self).__init__()
        self.data_dir = data_dir
        self.disease = disease
        self.device = device
        self.scale = scale
        self.data = data

        self._raw_data = self.load_disease_data(disease, data_dir, data)

        abundance = self._raw_data['abundance'].iloc[209:, :].T.to_numpy().astype(np.float)
        if scale:
            ab_scaler = StandardScaler()
            abundance = ab_scaler.fit_transform(abundance)
        abundance = torch.tensor(abundance, dtype=torch.float32)
        self.abundance = abundance.to(device)

        markers = self._raw_data['marker'].iloc[209:, :].T.to_numpy().astype(np.float)
        markers = torch.tensor(markers, dtype=torch.float32)
        self.markers = markers.to(device)

        labels = self._get_labels()
        self.labels_gpu = labels
        self.labels = labels.cpu().numpy().squeeze().astype('int')
        self.patients_ids = np.array([i for i in range(len(labels))])

    def __getitem__(self, idx):
        return self.abundance[idx], self.markers[idx], self.labels_gpu[idx]

    def __len__(self):
        return len(self.patients_ids)

    @staticmethod
    def load_disease_data(disease, data_dir, data):
        """
        Load the data of a specified disease.
        """
        raw_data = dict()

        if data == 'joint':
            raw_data['abundance'] = pd.read_csv(
                os.path.join(data_dir, data, 'abundance/abundance_{}.txt'.format(disease))
            )
            raw_data['marker'] = pd.read_csv(
                os.path.join(data_dir, data, 'marker/marker_{}.txt'.format(disease))
            )
            # fix the name of the first data column
            raw_data['abundance'] = raw_data['abundance'].rename(columns={'Unnamed: 0': 'sampleID'})
            raw_data['marker'] = raw_data['marker'].rename(columns={'Unnamed: 0': 'sampleID'})
        else:
            raw_data['abundance'] = pd.read_csv(
                os.path.join(data_dir, 'default', 'abundance/abundance_{}.txt'.format(disease)),
                sep="\t",
                skiprows=1
            )
            raw_data['marker'] = pd.read_csv(
                os.path.join(data_dir, data, 'marker/marker_{}.txt'.format(disease)),
                sep="\t",
                skiprows=1
            )

        # drop non-patient column
        raw_data['abundance'] = raw_data['abundance'].set_index('sampleID')
        raw_data['marker'] = raw_data['marker'].set_index('sampleID')

        assert len(raw_data['abundance'].columns) == len(raw_data['marker'].columns)
        return raw_data

    def _get_labels(self):
        """
        Create labels.
        """
        labels = np.array([[label_dict[i]] for i in self._raw_data['marker'].loc['disease'].to_list()])
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(self.device)
        return labels

    def train_test_split(self, test_size, random_state):
        """
        Do stratified train/test split.
        """
        train_ids, test_ids, y_train, y_test = model_selection.train_test_split(
            self.patients_ids,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        return train_ids, test_ids, y_train, y_test


class FullMicrobiomeDataset(MicrobiomeDataset):
    """
    Create dataset for all diseases - except for the target disease i.e. the `disease` argument.
    This dataset class is meant to be used for transfer learning only.
    """
    def __init__(self, data_dir, disease, device, scale=True, data='default', diseases=datasets):
        super(FullMicrobiomeDataset, self).__init__(data_dir, disease, device, scale, data, diseases)
        self.data_dir = data_dir
        self.disease = disease
        self.diseases = diseases
        self.device = device
        self.data = data

        self._raw_data = self.load_disease_data(disease, data_dir, data)

        abundance_stack = None
        markers_stack = None
        labels_stack = None
        for d in diseases:
            if d != disease:
                self._raw_data = self.load_disease_data(d, data_dir, data)
                abundance = self._raw_data['abundance'].iloc[209:, :].T.to_numpy()
                markers = self._raw_data['marker'].iloc[209:, :].T.to_numpy().astype(np.float)
                labels = self._get_labels()

                if scale:
                    ab_scaler = StandardScaler()
                    abundance = ab_scaler.fit_transform(abundance)

                if abundance_stack is None or markers_stack is None:
                    abundance_stack = abundance
                    markers_stack = markers
                    labels_stack = labels
                else:
                    abundance_stack = np.concatenate((abundance_stack, abundance), axis=0)
                    markers_stack = np.concatenate((markers_stack, markers), axis=0)
                    labels_stack = torch.cat((labels_stack, labels), dim=0)

        abundance_stack = torch.tensor(abundance_stack, dtype=torch.float32)
        markers_stack = torch.tensor(markers_stack, dtype=torch.float32)

        self.abundance = abundance_stack.to(device)
        self.markers = markers_stack.to(device)
        self.labels_gpu = labels_stack
        self.labels = labels_stack.cpu().numpy().squeeze().astype('int')

        self.patients_ids = np.array([i for i in range(len(self.abundance))])


class CustomDataset(Dataset):
    """
    Create dataset for a custom disease dataset.
    """
    def __init__(self, data_dir, disease, device, scale=True):
        """
        :param data_dir: the path to the raw data
        :param disease: the specific dataset
        :param device: on which device the data should be loaded
        :param scale: use StandardScaler on abundance
        """
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.disease = disease
        self.device = device

        self._raw_data = self.load_disease_data(disease, data_dir)

        abundance = self._raw_data['abundance'].iloc[1:, :].T.to_numpy().astype(np.float)
        if scale:
            ab_scaler = StandardScaler()
            abundance = ab_scaler.fit_transform(abundance)
        abundance = torch.tensor(abundance, dtype=torch.float32)
        self.abundance = abundance.to(device)

        markers = self._raw_data['marker'].iloc[1:, :].T.to_numpy().astype(np.float)
        markers = torch.tensor(markers, dtype=torch.float32)
        self.markers = markers.to(device)

        labels = self._get_labels()
        self.labels_gpu = labels
        self.labels = labels.cpu().numpy().squeeze().astype('int')
        self.patients_ids = np.array([i for i in range(len(labels))])

    def __getitem__(self, idx):
        return self.abundance[idx], self.markers[idx], self.labels_gpu[idx]

    def __len__(self):
        return len(self.patients_ids)

    @staticmethod
    def load_disease_data(disease, data_dir):
        """
        Load the data of a specified disease.
        """
        raw_data = dict()

        raw_data['abundance'] = pd.read_csv(
            os.path.join(data_dir, 'abundance/abundance_{}.txt'.format(disease)),
            sep="\t"
        )
        raw_data['marker'] = pd.read_csv(
            os.path.join(data_dir, 'marker/marker_{}.txt'.format(disease)),
            sep="\t"
        )

        # drop non-patient column
        raw_data['abundance'] = raw_data['abundance'].set_index('sampleID')
        raw_data['marker'] = raw_data['marker'].set_index('sampleID')

        assert len(raw_data['abundance'].columns) == len(raw_data['marker'].columns)
        return raw_data

    def _get_labels(self):
        """
        Create labels.
        """
        labels = np.array([[label_dict[i]] for i in self._raw_data['marker'].loc['disease'].to_list()])
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(self.device)
        return labels

    def train_test_split(self, test_size, random_state):
        """
        Do stratified train/test split.
        """
        train_ids, test_ids, y_train, y_test = model_selection.train_test_split(
            self.patients_ids,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        return train_ids, test_ids, y_train, y_test


class MetabolicDataset(MicrobiomeDataset):
    """
    A dataset class for the experiments with abundance, markers and metabolite modalities.
    """
    def __init__(self, data_dir, disease, device, scale=True, data='default', diseases=datasets):
        super(MetabolicDataset, self).__init__(data_dir, disease, device, scale, data, diseases)

        metabolite = pd.read_csv(
            os.path.join(self.data_dir, 'default', 'metabolite/metabolite_{}.txt'.format(self.disease)),
            sep="\t",
            skiprows=1
        )

        metabolite = metabolite.set_index('sampleID')
        metabolite = metabolite.iloc[209:, :].T.to_numpy().astype(np.float)

        if self.scale:
            metabolite_scaler = StandardScaler()
            metabolite = metabolite_scaler.fit_transform(metabolite)
        metabolite = torch.tensor(metabolite, dtype=torch.float32)

        self.metabolite = metabolite.to(self.device)

    def __getitem__(self, idx):
        return self.abundance[idx], self.markers[idx], self.metabolite[idx], self.labels_gpu[idx]
