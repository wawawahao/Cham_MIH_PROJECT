"""
1. 修改 MIH-Index 索引结构
2. 使用 xxhash 替代 sha256
3. 分为三步
    1) 检索结果
    2) 预处理验证
    3) 结果验证
3. 在 try_1.py 基础上做修改
    1) 修改 MIHIndex 结构, 每个 element 记录其所属 leaf_node 的标号, 以及在当前 leaf_node 下的位置信息
"""

import xxhash
import os
import pickle
import time
import numpy as np

from bitarray import bitarray
from tqdm import tqdm
from collections import deque

class Element:
    def __init__(self, id, hash):
        """
        id: 唯一标识符
        hash: 图像特征值
        merkle_hash: 使用 xxhash 根据 hash 计算得到
        leaf_pos: 记录当前 element 属于每个 bucket 的第几个叶子节点
        """
        self.id = id
        self.hash = hash
        self.merkle_hash = xxhash.xxh128(hash).digest()
        self.leaf_pos = []

class LeafNode:
    def __init__(self, id, elements):
        """
        id: 唯一标识符
        elements: 该 leaf_node 包含的 element 列表
        merkle_hash: 使用 xxhash 根据 elements 的 merkle_hash 拼接结果计算得到
        """
        self.id = id
        self.elements = elements
        self.merkle_hash = xxhash.xxh128(b''.join([element.merkle_hash for element in elements])).digest()

class BucketNode:
    def __init__(self, id, hashs, leaf_nodes):
        """
        id: 唯一标识符
        hashs: 当前 bucket 包含的特征值的全部情况(特征值子串的全部情况)
        leaf_nodes: 每一种情况对应一个 leaf_node
        hash2leaf: hashs 与 leaf_nodes 的映射关系
        merkle_hash: 使用 xxhash 根据 leaf_nodes 的 merkle_hash 拼接结果计算得到
        """
        self.id = id
        self.hashs = hashs
        self.leaf_nodes = leaf_nodes
        self.hash2leaf = dict(zip(hashs, leaf_nodes))
        self.merkle_hash = xxhash.xxh128(b''.join([leaf_node.merkle_hash for leaf_node in leaf_nodes])).digest()

class MIHIndex:
    def __init__(self, database_path = 'ada_total_hashes.bin', hash_length = 128, word_length = 16):
        """
        _hash_length: 图像特征值长度
        _word_length: 子串长度
        elements: 全部 element 列表
        buckets: 全部 bucket 列表
        root_merkle_hash: 根节点的 merkle_hash
        cnt: 计数器
        """
        self._hash_length = hash_length
        self._word_length = word_length
        self.elements = []
        self.buckets = []
        self.root_merkle_hash = None
        self.cnt = 0

        self.build_index(database_path)

    def build_index(self, database_path):
        """
        根据 database_path 提供的 .bin 文件构造 MIHIndex 索引结构
        """
        # 如果系统中存在 mih_index.pkl 文件, 那么直接 pickle 加载
        if os.path.exists('mih_index.pkl'):
            print(f"[INFO] 从系统中加载 mih_index.pkl 文件")
            with open('mih_index.pkl', 'rb') as f:
                mih = pickle.load(f)
                self.elements = mih.elements
                self.buckets = mih.buckets
                self.root_merkle_hash = mih.root_merkle_hash
                self.cnt = mih.cnt
                return
        # 如果系统中不存在 mih_index.pkl 文件, 那么需要重新构造
        else:
            print(f"[INFO] 系统中不存在 mih_index.pkl 文件, 需要重新构造")
            # 加载图像特征值数据库
            if database_path.endswith('.bin'):
                database = bitarray()
                with open(database_path, 'rb') as f:
                    database.fromfile(f)
            elif database_path.endswith('.t'):
                import torch
                t = torch.load(database_path)
                # 把 (-1, 1) 改成 (0, 1)
                t = (t + 1) // 2
                t = t.byte()
                database = bitarray(t.flatten().tolist())

            # 构造 elements
            len_database = len(database) // self._hash_length
            for element_id in tqdm(range(len_database), desc = '加载数据库'):
                element_hash = database[element_id * self._hash_length : (element_id + 1) * self._hash_length].tobytes()
                self.elements.append(Element(element_id, element_hash))
                self.cnt += 1
    
            # 构造 buckets
            buckets_count = self._hash_length // self._word_length
            for bucket_id in tqdm(range(buckets_count), desc = '构造 MIHIndex'):
                index = {}
                for element_id in range(len(self.elements)):
                    element_hash = bitarray()
                    element_hash.frombytes(self.elements[element_id].hash)
                    word = element_hash[bucket_id * self._word_length : (bucket_id + 1) * self._word_length].tobytes()
                    if word not in index:
                        index[word] = [element_id]
                    else:
                        index[word].append(element_id)
                hashs = []
                leaf_nodes = []
                for key, value in index.items():
                    hashs.append(key)
                    # 修改 element 存储 leaf_node 的标号逻辑
                    leaf_elements = []
                    cnt = 0
                    for i in value:
                        self.elements[i].leaf_pos.append((len(leaf_nodes), cnt))
                        cnt += 1
                        leaf_elements.append(self.elements[i])
                    leaf_nodes.append(LeafNode(len(leaf_nodes), leaf_elements))
                self.buckets.append(BucketNode(bucket_id, hashs, leaf_nodes))
            
            # 构造 root_merkle_hash
            self.root_merkle_hash = xxhash.xxh128(b''.join([bucket.merkle_hash for bucket in self.buckets])).digest()

            # 记录辅助信息, 即每个 element 属于 leaf_node 的标号
            # for bucket_id, bucket in enumerate(self.buckets):
            #     for leaf_id, leaf_node in enumerate(bucket.leaf_nodes):
            #         for element in leaf_node.elements:
            #             element.leaf_pos.append(leaf_node.id)
            
            # 写入 .pkl 文件
            with open('mih_index.pkl', 'wb') as f:
                pickle.dump(self, f)
        
        return
    
    @staticmethod
    def get_candidates(word, distance=2):
        """
        根据 query 和 distance 寻找可能的 candidates
        """
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
        """
        计算汉明距离
        """
        if type(a) == bytes:
            tmp = bitarray()
            tmp.frombytes(a)
            a = tmp
        if type(b) == bytes:
            tmp = bitarray()
            tmp.frombytes(b)
            b = tmp
        assert len(a) == len(b), "Bitarray must be of the same length"
        
        return (a ^ b).count(1)
    
    def linear_search(self, anchor, hamming_distance_threshold = 8):
        """
        根据 query 线性检索
        """
        if type(anchor) == bytes:
            tmp = bitarray()
            tmp.frombytes(anchor)
            anchor = tmp
        results = {}
        for element in self.elements:
            distance = self.hamming_distance(anchor, element.hash)
            if distance <= hamming_distance_threshold:
                results[element.id] = distance
        
        return results

    def query(self, anchor, hamming_distance_threshold = 8):
        """
        根据 query 基于 MIHIndex 索引结构检索
        """
        if type(anchor) == bytes:
            tmp = bitarray()
            tmp.frombytes(anchor)
            anchor = tmp
        if len(anchor) != self._hash_length:
            raise ValueError('Invalid hash length encountered, expected: {}, got: {}'.format(self._hash_length, len(anchor)))
        
        results = {}

        for bucket_id, bucket in enumerate(self.buckets):
            anchor_word = anchor[bucket_id * self._word_length : (bucket_id + 1) * self._word_length]
            suitable_words = self.get_candidates(anchor_word, hamming_distance_threshold // len(self.buckets))
            for word in suitable_words:
                if word in bucket.hash2leaf:
                    for element in bucket.hash2leaf[word].elements:
                        if element.id not in results:
                            distance = self.hamming_distance(anchor, element.hash)
                            if distance <= hamming_distance_threshold:
                                result = {
                                    'hash': element.hash,
                                    'distance': distance,
                                    'leaf_id': element.leaf_pos[0][0],
                                    'leaf_pos': element.leaf_pos[0][1],
                                }
                                results[element.id] = result

        return results
    
    def pre_verify(self, results):
        """
        对结果验证的 proof 做预处理
        """
        merkle_tree = {
            'root_merkle_hash': self.root_merkle_hash,
            'bucket_nodes_merkle_hash': b'',
            'leaf_nodes_merkle_hash': b'',
            'subling_merkle_hash': {},
        }
        leaf_res_id = {}
        for res in results:
            leaf_id = results[res]['leaf_id']
            leaf_pos = results[res]['leaf_pos']
            if leaf_id not in leaf_res_id:
                leaf_res_id[leaf_id] = [leaf_pos]
            else:
                leaf_res_id[leaf_id].append(leaf_pos)
            
        
        # print(leaf_res_id)
        
        # 预先计算 bucket_node 的 merkle_hash, 默认需要恢复第一个 bucket_node 的 merkle_hash
        for bucket_id, bucket in enumerate(self.buckets):
            if bucket_id == 0:
                merkle_tree['bucket_nodes_merkle_hash'] += b'\x00' * 16
            else:
                merkle_tree['bucket_nodes_merkle_hash'] += bucket.merkle_hash
        
        # 预先计算第一个 bucket_node 下的 leaf_node 的 merkle_hash
        for leaf_id, leaf_node in enumerate(self.buckets[0].leaf_nodes):
            if leaf_id not in leaf_res_id:
                merkle_tree['leaf_nodes_merkle_hash'] += leaf_node.merkle_hash
            else:
                merkle_tree['leaf_nodes_merkle_hash'] += b'\x00' * 16
                element_merkle_hash = b''
                # 预先计算当前 leaf_node 下的 element 的 merkle_hash
                for element in leaf_node.elements:
                    if element.id not in results:
                        element_merkle_hash += element.merkle_hash
                    else:
                        element_merkle_hash += b'\x00' * 16
                merkle_tree['subling_merkle_hash'][leaf_id] = element_merkle_hash

        return merkle_tree

def verify_proof(results, merkle_tree):
    """
    根据预处理的 proof 进行验证
    """
    for res in results:
        leaf_id = results[res]['leaf_id']
        leaf_pos = results[res]['leaf_pos']
        tmp_merkle_hash = bytearray(merkle_tree['subling_merkle_hash'][leaf_id])
        tmp_merkle_hash[leaf_pos * 16 : (leaf_pos + 1) * 16] = xxhash.xxh128(results[res]['hash']).digest()
        merkle_tree['subling_merkle_hash'][leaf_id] = bytes(tmp_merkle_hash)
    
    tmp_leaf_merkle_hash = bytearray(merkle_tree['leaf_nodes_merkle_hash'])
    for res in results:
        leaf_id = results[res]['leaf_id']
        tmp_leaf_merkle_hash[leaf_id * 16 : (leaf_id + 1) * 16] = xxhash.xxh128(merkle_tree['subling_merkle_hash'][leaf_id]).digest()
    merkle_tree['leaf_nodes_merkle_hash'] = bytes(tmp_leaf_merkle_hash)

    tmp_bucket_merkle_hash = bytearray(merkle_tree['bucket_nodes_merkle_hash'])
    tmp_bucket_merkle_hash[0 : 16] = xxhash.xxh128(merkle_tree['leaf_nodes_merkle_hash']).digest()
    merkle_tree['bucket_nodes_merkle_hash'] = bytes(tmp_bucket_merkle_hash)

    root_merkle_hash = xxhash.xxh128(merkle_tree['bucket_nodes_merkle_hash']).digest()

    return root_merkle_hash == merkle_tree['root_merkle_hash']

if __name__ == '__main__':
    # 构造 MIHIndex 
    mih = MIHIndex(database_path = 'ada_total_hashes.bin', hash_length = 128, word_length = 16)

    # 构造 query
    anchor = np.random.choice(mih.elements)
    # anchor = mih.elements[0]
    hamming_distance = 28
    print(f"[INFO] 待检索的 anchor: {anchor.hash.hex()}")
    # print(f"[INFO] anchor 所属的 leaf_node 标号: {anchor.leaf_pos}")
    print(f"[INFO] 汉明距离阈值: {hamming_distance}")

    # 线性检索
    s = time.time()
    results_linear = mih.linear_search(anchor.hash, hamming_distance)
    e = time.time() - s
    print(f"[INFO] 线性检索耗时: {e} 秒")
    print(f"[INFO] 线性检索结果: {list(results_linear.keys())}")

    # 基于 MIHIndex 检索
    s = time.time()
    results_query = mih.query(anchor.hash, hamming_distance)
    e = time.time() - s
    print(f"[INFO] MIH 检索耗时: {e} 秒")
    print(f"[INFO] MIH 检索结果: {list(results_query.keys())}")

    # 对检索结果的 proof 做预处理
    s = time.time()
    merkle_tree = mih.pre_verify(results_query)
    e = time.time() - s
    print(f"[INFO] 对检索结果的 proof 预处理耗时: {e} 秒")

    # 结果验证
    s = time.time()
    flag = verify_proof(results_query, merkle_tree)
    e = time.time() - s
    print(f"[INFO] 验证耗时: {e} 秒")
    print(f"[INFO] 验证结果: {flag}")