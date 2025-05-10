import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp
from scipy.spatial.distance import cdist
import multiprocessing as mp
import random
import math
import scipy.sparse as sp
from collections import defaultdict

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info(
            "Sample num: "
            + str(self.idx.shape[0])
            + ", Batch num: "
            + str(self.num_batch)
        )

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon

    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind:end_ind, ...]

                x_shape = (
                    len(idx_ind),
                    self.seq_len,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                x_shared = mp.RawArray("f", int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype="f").reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray("f", int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype="f").reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = (
                        start_index + chunk_size if i < num_threads - 1 else array_size
                    )
                    thread = threading.Thread(
                        target=self.write_to_shared_array,
                        args=(x, y, idx_ind, start_index, end_index),
                    )
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    if args.target:
        ptr = np.load(os.path.join(data_path, args.target, "his_initial.npz"))
    else:
        ptr = np.load(os.path.join(data_path, args.years, "his_initial.npz"))
    logger.info("Data shape: " + str(ptr["data"].shape))

    dataloader = {}
    for cat in ["train", "val", "test", "shift"]:
        idx = np.load(os.path.join(data_path, args.years, "idx_" + cat + ".npy"))
        dataloader[cat + "_loader"] = DataLoader(
            ptr["data"][..., : args.input_dim],
            idx,
            args.seq_len,
            args.horizon,
            args.bs,
            logger,
        )

    scaler = StandardScaler(mean=ptr["mean"], std=ptr["std"])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + "/data/"
    d = {
        "Bike_Chicago": [
            base_dir + "Bike_Chicago",
            base_dir + "Bike_Chicago/adj.npy",
            585,
        ],
        "Bus_NYC": [base_dir + "Bus_NYC", base_dir + "Bus_NYC/adj.npy", 226],
        "Taxi_Chicago": [
            base_dir + "Taxi_Chicago",
            base_dir + "Taxi_Chicago/adj.npy",
            77,
        ],
        "Taxi_NYC": [base_dir + "Taxi_NYC", base_dir + "Taxi_NYC/adj.npy", 263],
        "Bike_DC": [base_dir + "Bike_DC", base_dir + "Bike_DC/adj.npy", 532],
        "Bike_NYC": [base_dir + "Bike_NYC", base_dir + "Bike_NYC/adj.npy", 820],
        "PEMS04": [base_dir + "PEMS04", base_dir + "PEMS04/adj.npy", 307],
        "PEMS07": [base_dir + "PEMS07", base_dir + "PEMS07/adj.npy", 883],
        "PEMS03": [base_dir + "PEMS03", base_dir + "PEMS03/adj.npy", 358],
        "PEMS08": [base_dir + "PEMS08", base_dir + "PEMS08/adj.npy", 170],
        "Metro_NYC": [base_dir + "Metro_NYC", base_dir + "Metro_NYC/adj.npy", 428],
        "Ped_Melbourne": [
            base_dir + "Ped_Melbourne",
            base_dir + "Ped_Melbourne/adj.npy",
            55,
        ],
        "Speed_NYC": [base_dir + "Speed_NYC", base_dir + "Speed_NYC/adj.npy", 139],
        "Ped_Zurich": [base_dir + "Ped_Zurich", base_dir + "Ped_Zurich/adj.npy", 99],
        "311_NYC": [base_dir + "311_NYC", base_dir + "311_NYC/adj.npy", 71],
        "NewBike_Chicago": [
            base_dir + "NewBike_Chicago",
            base_dir + "NewBike_Chicago/adj.npy",
            609,
        ],
    }
    assert dataset in d.keys()
    return d[dataset]

class Spatial_Side_Information():
    def __init__(self, set_node_index, epsilon, c, seed, device):
        super(Spatial_Side_Information, self).__init__()
        self._anchor_node = set_node_index[0]
        self._node = np.concatenate((set_node_index[0], set_node_index[-1]))
        self._num_nodes = len(self._node)
        self._epsilon = epsilon
        self._c = c
        self._seed = seed
        self._device = device

    def get_random_anchorset(self):
        np.random.seed(self._seed)
        c = self._c
        n = len(self._anchor_node)
        distortion = math.ceil(np.log2(n))
        sampling_rep_rounds = c
        anchorset_num = sampling_rep_rounds * distortion
        anchorset_id = [np.array([]) for _ in range(anchorset_num)]
        for i in range(distortion):
            anchor_size = int(math.ceil(n / np.exp2(i + 1)))
            for j in range(sampling_rep_rounds):
                anchorset_id[i*sampling_rep_rounds+j] = np.sort(self._anchor_node[np.random.choice(n, size=anchor_size, replace=False)])
        return anchorset_id, anchorset_num

    def nodes_dist_range(self, adj, node_range):
        dists_dict = defaultdict(dict)
        if False:
            # pseudo Hamming distance
            for node in node_range:
                for neighbor in self._node:
                    if neighbor not in dists_dict[node]:
                        dists_dict[node][neighbor] = 0
                    diff = (abs(self._adj[node, self._node] - self._adj[neighbor, self._node]) >= self._epsilon)
                    if isinstance(adj, sp.coo_matrix):
                        dists_dict[node][neighbor] = sp.coo_matrix.sum(diff)
                    elif isinstance(adj, np.ndarray):
                        dists_dict[node][neighbor] = np.sum(diff)
                    if self._adj[node, neighbor] > 0:
                        dists_dict[node][neighbor] +=1
        else:
            for node in node_range:
                for neighbor in self._node:
                    if neighbor not in dists_dict[node]:
                        dists_dict[node][neighbor] = 0
                    diff = abs(adj[node, self._node] - adj[neighbor, self._node])
                    if isinstance(adj, sp.coo_matrix):
                        dists_dict[node][neighbor] = sp.coo_matrix.sum(diff)
                    elif isinstance(adj, np.ndarray):
                        dists_dict[node][neighbor] = np.sum(diff)
                    if adj[node, neighbor] > 0:
                        dists_dict[node][neighbor] += 1
        return dists_dict

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def all_pairs_dist_parallel(self, adj, num_workers=16):
        nodes = self._node
        if len(nodes) < 200:
            num_workers = int(num_workers/4)
        elif len(nodes) < 800:
            num_workers = int(num_workers/2)
        elif len(nodes) < 3000:
            num_workers = int(num_workers)
        slices = np.array_split(nodes, num_workers)
        pool = mp.Pool(processes = num_workers)
        results = [pool.apply_async(self.nodes_dist_range, args=(adj, slices[i], )) for i in range(num_workers)]
        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)

        pool.close()
        pool.join()
        return dists_dict


    # 计算hamming距离dict
    def precompute_dist_data(self, adj):
        dists_array = np.zeros((self._num_nodes, self._num_nodes))
        # 并行或者不并行
        dists_dict = self.all_pairs_dist_parallel(adj)
        # dists_dict = self.nodes_hamming_dist_range(self._node)
        for i, node in enumerate(self._node):
            #dists_array[i] = list(dists_dict[node].values())
            dists_array[i] = np.array(list(dists_dict[node].values()))
        return dists_array



    # 得到raw spatial side information
    def get_dist_min(self, adj=None, dist=None):
        anchorset_id, anchorset_num = self.get_random_anchorset()
        # print(anchorset_id)
        if dist is None:
            dist = self.precompute_dist_data(adj)
        # dist = self.all_pairs_hamming_dist_parallel()
        dist_min = torch.zeros(self._num_nodes, anchorset_num).to(self._device)
        dist_argmin = torch.zeros(self._num_nodes, anchorset_num).long().to(self._device)
        coefficient = torch.ones(self._num_nodes, anchorset_num).to(self._device)
        for j in enumerate(anchorset_id):
            jj = [np.where(self._node == element)[0][0] for element in j[-1]]
            for i, node_i in enumerate(self._node):
                dist_temp = dist[i, jj]
                dist_temp_tensor = torch.as_tensor(dist_temp, dtype=torch.float32)
                dist_min_temp, dist_argmin_temp = torch.min(dist_temp_tensor, dim=-1)
                dist_min[i, j[0]] = dist_min_temp
                dist_argmin[i, j[0]] = anchorset_id[j[0]][dist_argmin_temp]
        return dist_min, dist_argmin, coefficient

    def spatial_emb_matrix(self, adj=None, dist=None):
        dist_min, dist_argmin, coefficient = self.get_dist_min(adj, dist)
        spatial_emb_matrix = torch.mul(coefficient, dist_min)
        return spatial_emb_matrix, dist_argmin
    
    # dx: (b, o, n, n) -> dx: (n, n)
    def temporal_emb_matrix(self, adj=None, dist=None):
        dist_min, dist_argmin, coefficient = self.get_dist_min(adj, dist)
        temporal_emb_matrix = torch.mul(coefficient, dist_min)
        return temporal_emb_matrix, dist_argmin
    
class Spatial_Embedding():
    def __init__(self, num_nodes, adj, new_node_ratio, num_val, epsilon, c, seed, device, istest=False, test_ratio=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj = adj
        self.new_node_ratio = new_node_ratio
        self.num_val = num_val
        self.epsilon = epsilon
        self.c = c
        self.seed = seed
        self.device = device
        self.istest = istest
        self.test_ratio = test_ratio

    def dataset_node_segment(self):
        np.random.seed(self.seed)
        node_indices = np.arange(self.num_nodes)
        np.random.shuffle(node_indices)
        num_fixed_node = min(int(self.num_nodes / (1 + (self.num_val + 2) * self.new_node_ratio)), \
                             self.num_nodes - self.num_val - 2)
        num_additional_nodes_per = max(int(num_fixed_node * self.new_node_ratio), 1)
        fixed_indices = np.sort(node_indices[:num_fixed_node])
        train_indices = [fixed_indices, np.sort(node_indices[num_fixed_node:num_fixed_node + num_additional_nodes_per])]
        val_indices = {}
        for i in range(self.num_val):
            start_index = num_fixed_node + ((i+1) * num_additional_nodes_per)
            end_index = start_index + num_additional_nodes_per
        val_indices[i] = [fixed_indices, \
                          np.sort(node_indices[start_index:end_index])]
        test_indices = [fixed_indices, \
                        np.sort(node_indices[num_fixed_node + ((self.num_val+1) * num_additional_nodes_per) : num_fixed_node + ((self.num_val+2) * num_additional_nodes_per)])]
        
        indices = {
        'fixed': fixed_indices,
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
        }
        dataset_node_segment_index = {}
        for cat in ['fixed', 'train', 'val', 'test']:
            dataset_node_segment_index[cat + '_indices'] = indices[cat]
        
        return dataset_node_segment_index, num_fixed_node, num_additional_nodes_per

    def SSI_func(self, ob, dist):
        return Spatial_Side_Information(ob, self.epsilon, self.c, self.seed, self.device).spatial_emb_matrix(adj=self.adj, dist=dist)

    def load_node_index_and_segemnt(self):
        dataset_node_segment_index, num_ob, num_un = self.dataset_node_segment()
        node = {}
        sem = {}
        adj = {}
        node['fixed'] = dataset_node_segment_index['fixed' + '_indices']
        if self.istest == False:
            for cat in ['train', 'val', 'test']:
                if cat == 'val': 
                    if len(dataset_node_segment_index[cat + '_indices']) == 1:
                        node[cat] = dataset_node_segment_index[cat + '_indices'][0]
                        node[cat + '_observed_node'] = node[cat][0]
                        node[cat + '_unobserved_node'] = node[cat][-1]
                        node[cat + '_node'] = np.concatenate((node[cat + '_observed_node'], \
                                                    node[cat + '_unobserved_node']))
                        dist = cdist(self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                    self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                    metric='cityblock')
                        sem[cat], _ = self.SSI_func(node[cat], dist)
                        adj[cat] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                        adj[cat+'_observed'] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                        print('SEM for ' + cat + ' set has been calculated: ' + str(sem[cat].shape))
                    else:
                        for i in range(len(dataset_node_segment_index[cat])):
                            node[cat + '_' + str(i)] = dataset_node_segment_index[cat + '_indices'][i]
                            node[cat + '_observed_node_' + str(i)] = node[cat + '_' + str(i)][0]
                            node[cat + '_unobserved_node_' + str(i)] = node[cat + '_' + str(i)][-1]
                            node[cat + '_node_'] = np.concatenate((node[cat + '_observed_node'], \
                                                    node[cat + '_unobserved_node']))
                            dist = cdist(self.adj[node[cat + '_node' + str(i)], :][:, node[cat + '_node' + str(i)]], 
                                        self.adj[node[cat + '_node' + str(i)], :][:, node[cat + '_node' + str(i)]], 
                                        metric='cityblock')
                            adj[cat + '_' + str(i)] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                            adj[cat+'_observed' + '_' + str(i)] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                            sem[cat + '_' + str(i) + '_' + str(i)], _ = self.SSI_func(node[cat + '_' + str(i)], dist)
                else:
                    node[cat] = dataset_node_segment_index[cat + '_indices']
                    node[cat + '_observed_node'] = node[cat][0]
                    node[cat + '_unobserved_node'] = node[cat][-1]
                    node[cat + '_node'] = np.concatenate((node[cat + '_observed_node'], \
                                                        node[cat + '_unobserved_node']))
                    dist = cdist(self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                self.adj[node[cat + '_node'], :][:, node[cat + '_node']], 
                                metric='cityblock')
                    sem[cat], _ = self.SSI_func(node[cat], dist)
                    adj[cat] = self.adj[node[cat + '_node'], :][:,node[cat + '_node']]
                    adj[cat+'_observed'] = self.adj[node[cat + '_observed_node'], :][:, node[cat + '_observed_node']]
                    print('SEM for ' + cat + ' set has been calculated: ' + str(sem[cat].shape))
        else: 
            extra = dataset_node_segment_index['val_indices'][0]
            if self.test_ratio == 0.05:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = node['test'][-1][:len(node['test'][-1])//2]
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un//2
            elif self.test_ratio == 0.1:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un
            elif self.test_ratio == 0.15:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = np.concatenate((node['test'][-1], extra[-1][:len(node['test'][-1])//2]))
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un + num_un//2
            elif self.test_ratio == 0.2:
                node['test'] = dataset_node_segment_index['test_indices']
                node['test'][-1] = np.concatenate((node['test'][-1], extra[-1]))
                node['test_observed_node'] = node['test'][0]
                node['test_unobserved_node'] = node['test'][-1]
                node['test_node'] = np.concatenate((node['test_observed_node'], \
                                                    node['test_unobserved_node']))
                dist = cdist(self.adj[node['test_node'], :][:, node['test_node']], 
                                self.adj[node['test_node'], :][:, node['test_node']], 
                                metric='cityblock')
                sem['test'], _ = self.SSI_func(node['test'], dist)
                adj['test'] = self.adj[node['test_node'], :][:,node['test_node']]
                adj['test_observed'] = self.adj[node['test_observed_node'], :][:, node['test_observed_node']]
                num_un = num_un*2
        return node, sem, adj, num_ob, num_un