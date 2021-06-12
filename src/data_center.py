import pandas as pd
import datetime
import random
import numpy as np
import transformers as ppb
import torch
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp


class DataCenter(object):
    def __init__(self, device):
        super(DataCenter, self).__init__()
        self.sent_len = 200
        self.device = device

    def get_author(self, pid, pdata):
        return pdata.loc[pdata["P_id"] == pid, "Author"].tolist()[0]

    def get_win_id(self, dates, win_size):
        date_low = datetime.datetime.strptime('2018-07-01', '%Y-%m-%d')
        interval = (dates - date_low).days
        return int(interval / win_size)

    def load_dataSet(self, dataset, win_size, train_proportion):

        sample_loc = "F:\\experiment data\\%s\\samples.csv" % dataset
        history_loc = "F:\\experiment data\\%s\\history.csv" % dataset
        post_loc = "F:\\experiment data\\%s\\post.csv" % dataset

        bert_loc = 'F:\\huggingface models\\bert-base-chinese'
        tokenizer = ppb.BertTokenizer.from_pretrained(bert_loc)
        model = ppb.BertModel.from_pretrained(bert_loc).to(self.device)
        # tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-chinese')
        # model = ppb.BertModel.from_pretrained('bert-base-chinese').to(self.device)

        samples = pd.read_csv(sample_loc, sep=',', header=0)
        history = pd.read_csv(history_loc, sep='\001', header=0)
        post = pd.read_csv(post_loc, sep='\001', header=0)

        history['Date'] = pd.to_datetime(history['Date'], format="%Y-%m-%d")

        history['Win_id'] = 0
        if win_size == 'month':
            history.loc[:, 'Win_id'] = history['Date'].dt.month
        else:
            history.loc[:, 'Win_id'] = history.apply(lambda row: self.get_win_id(row['Date'], int(win_size)), axis=1)

        edges_windows = []
        adj_windows = []

        idx = list(set(history['Ori_user_id'].tolist() + history['Post_author'].tolist()))

        nodes_dim = len(list(bin(max(idx)))) - 2
        feature = np.zeros((len(idx), nodes_dim))
        for i, elem in enumerate(idx):
            feat = list(bin(elem))[2::]
            feature[i, 0:len(feat)] = np.array(feat, dtype=np.float32)

        grouped = history.groupby('Win_id')

        for name, group in grouped:
            graph = nx.Graph()
            for index, row in group.iterrows():
                graph.add_edge(row['Ori_user_id'], row['Post_author'])
            edges = np.array(graph.edges())
            edges_windows.append(edges)

        for edges in edges_windows:
            adj_windows.append(self.create_adj(idx, edges))

        feature = self.process_feature(feature).float().to(self.device)

        history = history.loc[history['Sample_label'] == 1]
        history['Content'] = history['Content'].astype(str)
        history.reset_index(drop=True, inplace=True)
        post['Title'] = post['Title'].astype(str)

        feats = self.bert_feature(history['Content'].tolist(), tokenizer, model)
        history['Feature'] = feats.numpy().tolist()
        feats = self.bert_feature(post['Title'].tolist(), tokenizer, model)
        post['Feature'] = feats.numpy().tolist()

        samples_ = samples.to_numpy()
        np.random.shuffle(samples_)
        num_sample = len(samples_)

        val_range = (1 - train_proportion) / 2
        train_pairs = samples_[0:int(train_proportion * num_sample)]
        val_pairs = samples_[int(train_proportion * num_sample): int((train_proportion + val_range) * num_sample)]
        test_pairs = samples_[int((train_proportion + val_range) * num_sample)::]

        np.random.shuffle(train_pairs)
        np.random.shuffle(val_pairs)
        np.random.shuffle(test_pairs)

        setattr(self, dataset + '_test', test_pairs)
        setattr(self, dataset + '_val', val_pairs)
        setattr(self, dataset + '_train', train_pairs)

        setattr(self, dataset + '_post', post)
        setattr(self, dataset + '_history', history)
        setattr(self, dataset + '_idx', idx)
        setattr(self, dataset + '_feature', feature)
        setattr(self, dataset + '_adj_windows', adj_windows)

    def bert_feature(self, sentences, tokenizer, model):
        feats = []
        batch_size = 96
        for batch, i in tqdm(enumerate(range(0, len(sentences), batch_size))):
            seq_len = min(batch_size, len(sentences) - i)
            encodings = tokenizer(sentences[i: i + seq_len], padding=True, max_length=512, truncation=True)
            input_ids = torch.tensor(encodings['input_ids']).to(self.device)
            attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                outputs = torch.mean(outputs.last_hidden_state, dim=1)
                feats.append(outputs.to('cpu'))
        feats = torch.cat(feats, dim=0)

        return feats

    def re_numbering(self, id_, all_users):
        return all_users.index(id_)

    def get_interval(self, date):
        return (datetime.datetime.strptime('2018-11-30', '%Y-%m-%d') - date).days

    def add_edge_attribute(self, row, e, e_attr):
        if e in e_attr.keys():
            if row['Date_Interval'] < e_attr[e]:
                e_attr[e] = row['Date_Interval']
        else:
            e_attr[e] = row['Date_Interval']

        return e_attr, row['Date_Interval']

    def list_to_array(self, token_list):
        if isinstance(token_list, float):
            return np.array([0], dtype=int)
        else:
            tokens = token_list.split(',')
            if len(tokens) > self.sent_len:
                tokens = random.sample(tokens, self.sent_len)
            return np.array(tokens, dtype=int)

    def create_adj(self, idx, edges_unordered):
        idx = np.array(idx, dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(edges_unordered, dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(idx.shape[0], idx.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        return adj.to(self.device)

    def process_feature(self, features):
        features = sp.csr_matrix(features, dtype=np.float32)
        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))

        return features

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

