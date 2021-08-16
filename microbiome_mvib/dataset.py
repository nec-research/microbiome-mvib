# coding=utf-8
#        Multimodal Variational Information Bottleneck
#
#   File:     dataset.py
#   Authors:  Filippo Grazioli filippo.grazioli@neclab.eu
#             Roman Siarheyeu raman.siarheyeu@neclab.eu
#             Giampaolo Pileggi giampaolo.pileggi@neclab.eu
#             Andrea Meiser andrea.meiser@neclab.eu
#
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#
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
