from bitarray import bitarray
from math import floor
import secrets
from tqdm import tqdm, trange
from hashlib import sha256
from collections import deque
import time
import json
import pickle
import os


class Element:
    def __init__(self, hash, id, raw_data=None):
        self.hash = hash
        self.id = id
        self.merkle_hash = sha256(hash).digest()


class LeafNode:
    def __init__(self, elements, id):
        self.id = id
        self.elements = elements
        self.merkle_hash = sha256(b''.join([element.merkle_hash for element in elements])).digest()


class BucketNode:
    def __init__(self, hashs, leaf_nodes, id):
        self.id = id
        self.hashs = hashs
        self.leaf_nodes = leaf_nodes
        self.hash2leaf = dict(zip(hashs, leaf_nodes))
        self.merkle_hash = sha256(b''.join([leaf_node.merkle_hash for leaf_node in leaf_nodes])).digest()


class MIHIndex:
    def __init__(self, database_path='5_23_total_small.bin', hash_length=128, word_length=16):
        self._hash_length = hash_length
        self._word_length = word_length

        self.elements = []
        self.buckets = []
        self.root_merkle_hash = None

        self.build_index(database_path)

    def build_index(self, database_path):
        # load MIHIndex
        if os.path.exists('mih_index.pkl'):
            with open('mih_index.pkl', 'rb') as f:
                mih = pickle.load(f)
                self.elements = mih.elements
                self.buckets = mih.buckets
                self.root_merkle_hash = mih.root_merkle_hash
                return
        else:
            print('MIHIndex not found, building MIHIndex from scratch...')
            if database_path.endswith('.bin'):
                database = bitarray()
                with open(database_path, 'rb') as f:
                    database.fromfile(f)
            elif database_path.endswith('.t'):
                import torch
                t = torch.load(database_path)
                # 把 (-1, 1) 改成 (0, 1)
                t = (t + 1) / 2
                t = t.byte()
                database = bitarray(t.flatten().tolist())
            len_database = len(database) // self._hash_length
            for element_id in tqdm(range(len_database), desc='Loading database'):
                element_hash = database[element_id*self._hash_length:(element_id+1)*self._hash_length].tobytes()
                self.elements.append(Element(element_hash, element_id))
            
            buckets_count = self._hash_length // self._word_length
            for bucket_id in tqdm(range(buckets_count), desc='Building index'):
                index = {}
                for element_id in range(len(self.elements)):
                    element_hash = bitarray()
                    element_hash.frombytes(self.elements[element_id].hash)
                    word = element_hash[bucket_id*self._word_length:(bucket_id+1)*self._word_length].tobytes()
                    if word not in index:
                        index[word] = [element_id]
                    else:
                        index[word].append(element_id)
                leaf_nodes = []
                hashs = []
                for key, value in index.items():
                    leaf_nodes.append(LeafNode([self.elements[i] for i in value], len(leaf_nodes)))
                    hashs.append(key)
                self.buckets.append(BucketNode(hashs, leaf_nodes, bucket_id))
            self.root_merkle_hash = sha256(b''.join([bucket.merkle_hash for bucket in self.buckets])).digest()
            # dump MIHIndex
            with open('mih_index.pkl', 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def get_candidates(word, distance=2):
        entries = set()
        queue = deque([(word.copy(), 0, distance)])  # 使用deque存储（当前位串，当前处理位置，剩余汉明距离）

        while queue:
            current_word, position, remaining_distance = queue.popleft()
            if position == len(word):
                entries.add(current_word.tobytes())
                continue

            # 处理当前位，不改变当前位
            queue.append((current_word.copy(), position + 1, remaining_distance))

            if remaining_distance > 0:
                # 改变当前位
                flipped_word = current_word.copy()
                flipped_word[position] = not current_word[position]
                queue.append((flipped_word, position + 1, remaining_distance - 1))

        return entries

    @staticmethod
    def hamming_distance(a, b):
        if type(a) == bytes:
            tmp = bitarray()
            tmp.frombytes(a)
            a = tmp
        if type(b) == bytes:
            tmp = bitarray()
            tmp.frombytes(b)
            b = tmp
        assert len(a) == len(b), "Bitarrays must be of the same length"
        return (a ^ b).count(1)
    
    def linear_search(self, anchor, hamming_distance_threshold=8):
        if type(anchor) == bytes:
            tmp = bitarray()
            tmp.frombytes(anchor)
            anchor = tmp
        results = {}
        for i, element in enumerate(self.elements):
            distance = self.hamming_distance(anchor, element.hash)
            if distance <= hamming_distance_threshold:
                results[element.id] = distance
        return results

    def query(self, anchor, hamming_distance_threshold=8):
        """
        Query index for candidates pre-set hamming distance threshold
        :param anchor: Candidate hash
        :param hamming_distance_threshold: Hamming distance threshold
        :return: List of candidates
        """
        if type(anchor) == bytes:
            tmp = bitarray()
            tmp.frombytes(anchor)
            anchor = tmp
        if len(anchor) != self._hash_length:
            raise ValueError('Invalid hash length encountered, expected: {}, got: {}'.format(self._hash_length, len(anchor)))

        results = {}

        for bucket_id, bucket in enumerate(self.buckets):
            anchor_word = anchor[bucket_id*self._word_length:(bucket_id+1)*self._word_length]
            suitable_words = self.get_candidates(anchor_word, hamming_distance_threshold // len(self.buckets))
            for word in suitable_words:
                if word in bucket.hash2leaf:
                    for element in bucket.hash2leaf[word].elements:
                        if element.id not in results:
                            distance = self.hamming_distance(anchor, element.hash)
                            if distance <= hamming_distance_threshold:
                                result = {
                                    'bucket_id': bucket.id,
                                    'leaf_id': bucket.hash2leaf[word].id,
                                    'distance': distance,
                                    'hash': element.hash.hex(),
                                    'subling_merkle_hashs': []
                                }
                                for subling in bucket.hash2leaf[word].elements:
                                    if subling.id != element.id:
                                        result['subling_merkle_hashs'].append(subling.merkle_hash.hex())
                                    else:
                                        result['subling_merkle_hashs'].append(None)
                                results[element.id] = result

        merkle_tree = {
            'root': self.root_merkle_hash.hex(),
            'buckets': None,
            'leaves': None,
        }

        buckets_count = [0 for _ in range(len(self.buckets))]
        leaves_count = []
        for bucket in self.buckets:
            leaves_count_i = [0 for _ in range(len(bucket.leaf_nodes))]
            leaves_count.append(leaves_count_i)
        
        for result in results.values():
            buckets_count[result['bucket_id']] += 1
            leaves_count[result['bucket_id']][result['leaf_id']] += 1
        
        merkle_tree['buckets'] = [None for _ in range(len(self.buckets))]
        for i, bucket in enumerate(self.buckets):
            if buckets_count[i] == 0:
                merkle_tree['buckets'][i] = bucket.merkle_hash.hex()
        
        merkle_tree['leaves'] = [None for _ in range(len(self.buckets))]
        for i, bucket in enumerate(self.buckets):
            if buckets_count[i] == 0:
                merkle_tree['leaves'][i] = None
            else:
                merkle_tree['leaves'][i] = []
                for j, leaf in enumerate(bucket.leaf_nodes):
                    if leaves_count[i][j] == 0:
                        merkle_tree['leaves'][i].append(leaf.merkle_hash.hex())
                    else:
                        merkle_tree['leaves'][i].append(None)
        
        return results, merkle_tree


def verify_proof(results, merkle_tree):
    bucket_hash = []
    try:  # 当敌手增加或删除了 results 以及 merkle_tree 时，可能会导致异常
        for i, bucket in enumerate(merkle_tree['buckets']):
            if bucket is not None:
                bucket_hash.append(bytes.fromhex(merkle_tree['buckets'][i]))
            else:
                # 利用叶子节点的 hash 恢复 bucket hash
                leaf_hash = []
                for j, leaf in enumerate(merkle_tree['leaves'][i]):
                    if leaf is not None:
                        leaf_hash.append(bytes.fromhex(leaf))
                    else:
                        # 利用检索结果恢复叶子节点 hash
                        # 根据 bucket_id, leaf_id 找到检索结果
                        result = None
                        for result in results.values():
                            if result['bucket_id'] == i and result['leaf_id'] == j:
                                break
                        result_hash = sha256(bytes.fromhex(result['hash'])).digest()
                        # 根据 result_hash 和 subling_merkle_hashs 恢复 leaf hash
                        subling_merkle_hashs = []
                        for subling_merkle_hash in result['subling_merkle_hashs']:
                            if subling_merkle_hash is not None:
                                subling_merkle_hashs.append(bytes.fromhex(subling_merkle_hash))
                            else:
                                subling_merkle_hashs.append(result_hash)
                        leaf_hash.append(sha256(b''.join(subling_merkle_hashs)).digest())
                bucket_hash.append(sha256(b''.join(leaf_hash)).digest())
        root_hash = sha256(b''.join(bucket_hash)).digest()
    except:
        return False
    return root_hash == bytes.fromhex(merkle_tree['root'])


def random_hash(length):
    return secrets.token_bytes(length // 8)


if __name__ == '__main__':
    mih = MIHIndex(database_path='webface10b.bin', hash_length=128, word_length=16)
    anchor = bytes.fromhex("c595f84f4eaf47f46bc1186098802f90")
    print('待检索的 anchor: ', anchor.hex())

    t = 8
    print('汉明距离阈值: ', t)
    s = time.time()
    results = mih.linear_search(anchor, hamming_distance_threshold=t)
    print('线性搜索耗时: ', time.time() - s)
    print('线性搜索结果: ', list(results.keys()))

    s = time.time()
    results, merkle_tree = mih.query(anchor, hamming_distance_threshold=t)
    # print(json.dumps(results, indent=4))
    # print(merkle_tree)
    print('MIH 搜索结果: ', list(results.keys()))
    print('MIH 搜索耗时: ', time.time() - s)

    s = time.time()
    print('验证结果: ', verify_proof(results, merkle_tree))
    print('验证耗时: ', time.time() - s)
