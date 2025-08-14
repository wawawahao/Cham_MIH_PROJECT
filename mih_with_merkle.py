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

from charm.toolbox.Hash import ChamHash
from charm.toolbox.integergroup import IntegerGroupQ
from charm.core.math.integer import integer

import numpy as np

class ChamHash_Adm05(ChamHash):
    def __init__(self, p=0, q=0):
        ChamHash.__init__(self)
        global group
        group = IntegerGroupQ(0)
        group.p, group.q, group.r = p, q, 2
        self.group = group
    
    def paramgen(self, secparam=1024, datapath='cham_info.pkl'):
        if os.path.exists(datapath):
            with open(datapath, 'rb') as f:
                cham_info = pickle.load(f)
                pk = {}
                sk = {}
                pk['g'] = group.deserialize(cham_info['pk']['g'])
                pk['y'] = group.deserialize(cham_info['pk']['y'])
                sk['x'] = group.deserialize(cham_info['sk']['x'])
        else:
            if group.p == 0 or group.q == 0:
                group.paramgen(secparam)
            g, x = group.randomGen(), group.random()    # g, [1,q-1]
            y = g ** x
            pk = {'g': g, 'y': y}
            sk = {'x': x}
            pk_serialized = {
                'g': group.serialize(g),
                'y': group.serialize(y)
            }
            sk_serialized = {
                'x': group.serialize(x)
            }
            with open(datapath, 'wb') as f:
                cham_info = {
                    'pk': pk_serialized,
                    'sk': sk_serialized
                }
                pickle.dump(cham_info, f)
        return pk, sk
    
    def hash(self, pk, m, r=0, s=0):
        p, q = group.p, group.q
        if r == 0:
            r = group.random()
        if s == 0:
            s = group.random()

        if type(r) == bytes:
            r = group.deserialize(r)
        if type(s) == bytes:
            s = group.deserialize(s)

        e = group.hash(m, r)
        
        C = r - (((pk['y'] ** e) * (pk['g'] ** s)) % p) % q
        return group.serialize(C), group.serialize(r), group.serialize(s)
    
    @staticmethod
    def find_collision(pk, sk, C, new_message):
        p, q = group.p, group.q
        k_prime = group.random()
        C = group.deserialize(C)
        r_prime = C + ((pk['g'] ** k_prime) % p) % q
        e_prime = group.hash(new_message, r_prime)
        s_prime = (k_prime - (e_prime * sk['x'])) % q
        C_prime = r_prime - (((pk['y'] ** e_prime) * (pk['g'] ** s_prime)) % p) % q
        return group.serialize(C_prime), group.serialize(r_prime), group.serialize(s_prime)


class Element:
    def __init__(self, hash, id, raw_data=None):
        self.hash = hash
        self.id = id
        self.merkle_hash = sha256(hash).digest()


class LeafNode:
    def __init__(self, elements, id, cham_hash, pk):
        self.id = id
        self.elements = elements
        self.C = None
        self.r = None
        self.s = None
        # self.merkle_hash = sha256(b''.join([element.merkle_hash for element in elements])).digest()
        self.C, self.r, self.s = cham_hash.hash(pk, b''.join([element.merkle_hash for element in elements]))
        self.merkle_hash = sha256(self.C).digest()


class BucketNode:
    def __init__(self, hashs, leaf_nodes, id, cham_hash, pk):
        self.id = id
        self.hashs = hashs
        self.leaf_nodes = leaf_nodes
        self.hash2leaf = dict(zip(hashs, leaf_nodes))
        self.C = None
        self.r = None
        self.s = None
        # self.merkle_hash = sha256(b''.join([leaf_node.merkle_hash for leaf_node in leaf_nodes])).digest()
        self.C, self.r, self.s = cham_hash.hash(pk, b''.join([leaf_node.merkle_hash for leaf_node in leaf_nodes]))
        self.merkle_hash = sha256(self.C).digest()


class MIHIndex:
    def __init__(self, database_path='5_23_total_small.bin', hash_length=128, word_length=16, cham_hash=None, pk=None, sk=None):
        self._hash_length = hash_length
        self._word_length = word_length

        self.elements = []
        self.buckets = []
        self.root_merkle_hash = None
        self.cnt = 0

        self.build_index(database_path, cham_hash, pk)

    def build_index(self, database_path, cham_hash, pk):
        # load MIHIndex
        if os.path.exists('mih_index.pkl'):
            with open('mih_index.pkl', 'rb') as f:
                mih = pickle.load(f)
                self.elements = mih.elements
                self.buckets = mih.buckets
                self.root_merkle_hash = mih.root_merkle_hash
                self.cnt = mih.cnt
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
                self.cnt += 1
            
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
                    leaf_nodes.append(LeafNode([self.elements[i] for i in value], len(leaf_nodes), cham_hash, pk))
                    hashs.append(key)
                self.buckets.append(BucketNode(hashs, leaf_nodes, bucket_id, cham_hash, pk))
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
        # 根据当前element列表生成一个candidate列表，全部用0表示，当某一个element被作为candidate时，将其置为1，用于后续再次被选为candidate时，跳过
        candidate_element = [0 for i in range(len(self.elements))]
        # print(len(candidate_element))
        for bucket_id, bucket in enumerate(self.buckets):
            anchor_word = anchor[bucket_id*self._word_length:(bucket_id+1)*self._word_length]
            suitable_words = self.get_candidates(anchor_word, hamming_distance_threshold // len(self.buckets))
            for word in suitable_words:
                if word in bucket.hash2leaf:
                    for element in bucket.hash2leaf[word].elements:
                        if candidate_element[element.id] != 1:
                            candidate_element[element.id] = 1
                            if element.id not in results:
                                distance = self.hamming_distance(anchor, element.hash)
                                if distance <= hamming_distance_threshold:
                                    result = {
                                        'bucket_id': bucket.id,
                                        'leaf_id': bucket.hash2leaf[word].id,
                                        'distance': distance,
                                        'hash': element.hash.hex(),
                                        # 'subling_merkle_hashs': [],
                                        # 'leaf_r': bucket.hash2leaf[word].r,
                                        # 'leaf_s': bucket.hash2leaf[word].s,
                                        # 'bucket_r': bucket.r,
                                        # 'bucket_s': bucket.s
                                    }
                                    # for subling in bucket.hash2leaf[word].elements:
                                    #     if subling.id != element.id:
                                    #         result['subling_merkle_hashs'].append(subling.merkle_hash.hex())
                                    #     else:
                                    #         result['subling_merkle_hashs'].append(None)
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

        # for i, bucket in enumerate(self.buckets):
        #     if buckets_count[i] == 0:
        #         merkle_tree['buckets'][i] = bucket.merkle_hash.hex()
        
        """修改 返回检索结果 逻辑"""
        merkle_tree['buckets'] = {}
        for i, bucket in enumerate(self.buckets):
            if buckets_count[i] == 0:
                # 如果当前 bucket 没有检索结果, 那么直接返回 bucket 的 merkle_hash
                merkle_tree['buckets'][i] = {
                    'merkle_hash': bucket.merkle_hash.hex()
                }
            else:
                # 如果当前 bucket 有检索结果, 那么需要返回 r, s 信息来用于验证时得到 merkle_hash
                merkle_tree['buckets'][i] = {
                    'merkle_hash': None,
                    'bucket_r': bucket.r,
                    'bucket_s': bucket.s
                }

        
        # merkle_tree['leaves'] = [None for _ in range(len(self.buckets))]
        # for i, bucket in enumerate(self.buckets):
        #     if buckets_count[i] == 0:
        #         merkle_tree['leaves'][i] = None
        #     else:
        #         merkle_tree['leaves'][i] = []
        #         for j, leaf in enumerate(bucket.leaf_nodes):
        #             if leaves_count[i][j] == 0:
        #                 merkle_tree['leaves'][i].append(leaf.merkle_hash.hex())
        #             else:
        #                 merkle_tree['leaves'][i].append(None)

        """修改 返回检索结果 逻辑"""
        merkle_tree['leaves'] = {}
        for i, bucket in enumerate(self.buckets):
            if buckets_count[i] == 0:
                # 如果当前 bucket 没有检索结果, 那么刚刚的 merkle_tree['buckets'] 已经记录了 bucket 的 merkle_hash, 此处无需多余记录信息
                merkle_tree['leaves'][i] = None
            else:
                # 如果当前 bucket 有检索结果, 那么需要找到是哪个 leaf 的信息, 以及其余辅助验证信息
                merkle_tree['leaves'][i] = {}
                for j, leaf in enumerate(bucket.leaf_nodes):
                    if leaves_count[i][j] == 0:
                        # 检索结果不在当前叶子节点
                        merkle_tree['leaves'][i][j] = {
                            'merkle_hash': leaf.merkle_hash.hex()
                        }
                    else:
                        # 检索结果在当前叶子节点
                        merkle_tree['leaves'][i][j] = {
                            'merkle_hash': None,
                            'leaf_r': leaf.r,
                            'leaf_s': leaf.s,
                            'subling_merkle_hashs': []
                        }
                        for ele in leaf.elements:
                            if ele.id not in results:
                                # 该 element 不在检索结果中
                                merkle_tree['leaves'][i][j]['subling_merkle_hashs'].append(ele.merkle_hash.hex())
                            else:
                                if results[ele.id]['bucket_id'] == i and results[ele.id]['leaf_id'] == j:
                                    # 该 element 在检索结果中
                                    merkle_tree['leaves'][i][j]['subling_merkle_hashs'].append(None)

        return results, merkle_tree
    
    def add(self, element, cham_hash, pk, sk):
        if type(element) == bytes:
            tmp = bitarray()
            tmp.frombytes(element)
            anchor = tmp
        if len(anchor) != self._hash_length:
            raise ValueError('Invalid hash length encountered, expected: {}, got: {}'.format(self._hash_length, len(anchor)))

        for bucket in self.buckets:
            anchor_word = anchor[bucket.id*self._word_length:(bucket.id+1)*self._word_length].tobytes()
            if anchor_word in bucket.hash2leaf:
                # print("Option 1, need to add element to existing leaf node")
                leaf_node = bucket.hash2leaf[anchor_word]
                tmp_merkle_hash = b''.join([ele.merkle_hash for ele in leaf_node.elements])
                tmp_merkle_hash += sha256(anchor.tobytes()).digest()
                temp_hash, new_r, new_s = cham_hash.find_collision(pk, sk, leaf_node.C, tmp_merkle_hash)
                if sha256(temp_hash).digest() == leaf_node.merkle_hash:
                    # print("Find collision successfully, adding element to leaf node")
                    leaf_node.elements.append(Element(anchor.tobytes(), self.cnt))
                    leaf_node.r = new_r
                    leaf_node.s = new_s
                else:
                    raise ValueError('Failed to add element, merkle hash mismatch')
            else:
                # print("Option 2, need to create a new leaf node")
                new_leaf_node = LeafNode([Element(anchor.tobytes(), self.cnt)], len(bucket.leaf_nodes), cham_hash, pk)
                tmp_merkle_hash = b''.join([leaf.merkle_hash for leaf in bucket.leaf_nodes])
                tmp_merkle_hash += new_leaf_node.merkle_hash
                temp_hash, new_r, new_s = cham_hash.find_collision(pk, sk, bucket.C, tmp_merkle_hash)
                if sha256(temp_hash).digest() == bucket.merkle_hash:
                    # print("Find collision successfully, adding new leaf node")
                    bucket.leaf_nodes.append(new_leaf_node)
                    bucket.hashs.append(anchor_word)
                    bucket.hash2leaf[anchor_word] = new_leaf_node
                    bucket.r = new_r
                    bucket.s = new_s
                else:
                    raise ValueError('Failed to add element, merkle hash mismatch')

        self.elements.append(Element(anchor.tobytes(), self.cnt))
        self.cnt += 1

        return

    def delete(self, element, cham_hash, pk, sk):
        if type(element) == bytes:
            tmp = bitarray()
            tmp.frombytes(element)
            anchor = tmp
        if len(anchor) != self._hash_length:
            raise ValueError('Invalid hash length encountered, expected: {}, got: {}'.format(self._hash_length, len(anchor)))

        for bucket in self.buckets:
            anchor_word = anchor[bucket.id*self._word_length:(bucket.id+1)*self._word_length].tobytes()
            leaf_node = bucket.hash2leaf[anchor_word]
            tmp_merkle_hash = b''
            for ele in leaf_node.elements:
                if ele.merkle_hash != sha256(anchor.tobytes()).digest():
                    tmp_merkle_hash += ele.merkle_hash
            temp_hash, new_r, new_s = cham_hash.find_collision(pk, sk, leaf_node.C, tmp_merkle_hash)
            if sha256(temp_hash).digest() == leaf_node.merkle_hash:
                print("Find collision successfully, deleting element from leaf node")
                for ele in leaf_node.elements:
                    if ele.merkle_hash == sha256(anchor.tobytes()).digest():
                        leaf_node.elements.remove(ele)
                        break
                leaf_node.r = new_r
                leaf_node.s = new_s
            else:
                raise ValueError('Failed to delete element, merkle hash mismatch')

        for ele in self.elements:
            if ele.merkle_hash == sha256(anchor.tobytes()).digest():
                self.elements.remove(ele)

        return 

    def save(self, filepath='mih_index.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        return

"""修改 验证 逻辑"""
def verify_proof(results, merkle_tree, cham_hash, pk):
    bucket_hash = []
    # result_acc = [0 for i in range(len(results))]
    try:  # 当敌手增加或删除了 results 以及 merkle_tree 时，可能会导致异常
        for i, bucket in enumerate(merkle_tree['buckets']):
            # print(merkle_tree['buckets'][i]['merkle_hash'])
            if merkle_tree['buckets'][i]['merkle_hash'] is not None:
                # 如果当前 bucket 的 merkle_hash 存在, 那么不需要计算恢复
                bucket_hash.append(bytes.fromhex(merkle_tree['buckets'][i]['merkle_hash']))
            else:
                # 如果当前 bucket 的 merkle_hash 不存在, 那么需要利用 leaf 的 merkle_hash 恢复 bucket 的 merkle_hash
                leaf_hash = []
                for j, leaf in enumerate(merkle_tree['leaves'][i]):
                    # print(merkle_tree['leaves'][i][j]['merkle_hash'])
                    if merkle_tree['leaves'][i][j]['merkle_hash'] is not None:
                        leaf_hash.append(bytes.fromhex(merkle_tree['leaves'][i][j]['merkle_hash']))
                    else:
                        # 利用检索结果恢复叶子节点 hash
                        # 根据 bucket_id, leaf_id 找到检索结果
                        # temp_result = {}
                        # for i, result in enumerate(results):
                        #     if results[i]['bucket_id'] == i and results[i]['leaf_id'] == j:
                        #         temp_result[result] = result
                        # result_hash = sha256(bytes.fromhex(result['hash'])).digest()
                        # 根据 result_hash 和 subling_merkle_hashs 恢复 leaf hash
                        
                        subling_merkle_hashs = []
                        result_hash = []
                        for res in results:
                            if results[res]['bucket_id'] == i and results[res]['leaf_id'] == j:
                                result_hash.append(sha256(bytes.fromhex(results[res]['hash'])).digest())
                        cnt = 0
                        # print(merkle_tree['leaves'][i][j]['subling_merkle_hashs'])
                        for subling_merkle_hash in merkle_tree['leaves'][i][j]['subling_merkle_hashs']:
                            if subling_merkle_hash is not None:
                                subling_merkle_hashs.append(bytes.fromhex(subling_merkle_hash))
                            else:
                                result = None
                                # print(result)
                                # print(i, j)
                                # print(results)
                                # for res in results:
                                #     # print(f"res['bucket_id]: {res['bucket_id']}, res['leaf_id']: {res['leaf_id']}, res['flag']: {res['flag']}")
                                #     if results[res]['bucket_id'] == i and results[res]['leaf_id'] == j and results[res]['flag'] == False:
                                #         result_hash = results[res]['hash']
                                #         results[res]['flag'] = True
                                        
                                # print(f"[CHECK] result: {result}")
                                subling_merkle_hashs.append(result_hash[cnt])
                                cnt += 1
                        temp_hash_1, temp_r_1, temp_s_2 = cham_hash.hash(pk, b''.join(subling_merkle_hashs), merkle_tree['leaves'][i][j]['leaf_r'], merkle_tree['leaves'][i][j]['leaf_s'])
                        leaf_hash.append(sha256(temp_hash_1).digest())
                        
                temp_hash_2, temp_r_1, temp_r_2 = cham_hash.hash(pk, b''.join(leaf_hash), merkle_tree['buckets'][i]['bucket_r'], merkle_tree['buckets'][i]['bucket_s'])
                bucket_hash.append(sha256(temp_hash_2).digest())
        root_hash = sha256(b''.join(bucket_hash)).digest()
    except:
        return False
    return root_hash == bytes.fromhex(merkle_tree['root'])


def random_hash(length):
    return secrets.token_bytes(length // 8)


if __name__ == '__main__':

    # 初始化 chamhash 的参数p, q
    p = integer(141660875619984104245410764464185421040193281776686085728248762539241852738181649330509191671665849071206347515263344232662465937366909502530516774705282764748558934610432918614104329009095808618770549804432868118610669336907161081169097403439689930233383598055540343198389409225338204714777812724565461351567)
    q = integer(70830437809992052122705382232092710520096640888343042864124381269620926369090824665254595835832924535603173757631672116331232968683454751265258387352641382374279467305216459307052164504547904309385274902216434059305334668453580540584548701719844965116691799027770171599194704612669102357388906362282730675783)
    cham_hash = ChamHash_Adm05(p, q)
    pk, sk = cham_hash.paramgen()

    mih = MIHIndex(database_path='ada_total_hashes.bin', hash_length=128, word_length=16, cham_hash=cham_hash, pk=pk, sk=sk)
    anchor = bytes.fromhex("ffffffffffffffffffffffffffffffff")
    # 随机选取 element
    # anchor = np.random.choice(mih.elements).hash
    # anchor = mih.elements[0].hash
    print('待检索的 anchor: ', anchor.hex())

    t = 28
    print('汉明距离阈值: ', t)
    s = time.time()
    results = mih.linear_search(anchor, hamming_distance_threshold=t)
    print('线性搜索耗时: ', time.time() - s)
    print('线性搜索结果: ', list(results.keys()))

    s = time.time()
    results, merkle_tree = mih.query(anchor, hamming_distance_threshold=t)
    # print(json.dumps(results, indent=4))
    # print(merkle_tree)
    print('MIH 搜索耗时: ', time.time() - s)
    print('MIH 搜索结果: ', list(results.keys()))
    # 输出 results 和 merkle_tree
    # print('Merkle 树: ', merkle_tree)
    # print('搜索结果: ', results)

    s = time.time()
    print('验证结果: ', verify_proof(results, merkle_tree, cham_hash, pk))
    print('验证耗时: ', time.time() - s)

    """ 1. test add one """
    # new_element = bytes.fromhex("c595f84f4eaf47f46bc1186098802f91")
    # # new_element = bytes.fromhex("ffffffffffffffffffffffffffffffff")
    # print('待添加的 anchor: ', new_element.hex())
    # s = time.time()
    # mih.add(new_element, cham_hash, pk, sk)
    # print('添加耗时: ', time.time() - s)
    # mih.save()

    # print("汉明距离阈值： ", t)
    # s = time.time()
    # results = mih.linear_search(anchor, hamming_distance_threshold=t)
    # print('线性搜索耗时: ', time.time() - s)
    # print('线性搜索结果: ', list(results.keys()))

    # s = time.time()
    # results, merkle_tree = mih.query(anchor, hamming_distance_threshold=t)
    # # print(json.dumps(results, indent=4))
    # # print(merkle_tree)
    # print('MIH 搜索耗时: ', time.time() - s)
    # print('MIH 搜索结果: ', list(results.keys()))

    # s = time.time()
    # print('验证结果: ', verify_proof(results, merkle_tree, cham_hash, pk))
    # print('验证耗时: ', time.time() - s)

    # anchor_test = bytes.fromhex("c595f84f4eaf47f46bc1186098802f91")
    # print('测试 anchor: ', anchor_test.hex())
    # t = 8
    # print('汉明距离阈值: ', t)
    # s = time.time()
    # results = mih.linear_search(anchor_test, hamming_distance_threshold=t)
    # print('线性搜索耗时: ', time.time() - s)
    # print('线性搜索结果: ', list(results.keys()))

    # s = time.time()
    # results, merkle_tree = mih.query(anchor_test, hamming_distance_threshold=t)
    # print('MIH 搜索耗时: ', time.time() - s)
    # print('MIH 搜索结果: ', list(results.keys()))

    # s = time.time()
    # print('验证结果: ', verify_proof(results, merkle_tree, cham_hash, pk))
    # print('验证耗时: ', time.time() - s)

    """ 2. test delete one """
    # del_element = bytes.fromhex("c595f84f4eaf47f46bc1186098802f91")
    # # del_element = bytes.fromhex("ffffffffffffffffffffffffffffffff")
    # print('待删除的 anchor: ', del_element.hex())
    # s = time.time()
    # mih.delete(del_element, cham_hash, pk, sk)
    # print('删除耗时: ', time.time() - s)
    # mih.save()

    # print("汉明距离阈值： ", t)
    # s = time.time()
    # results = mih.linear_search(anchor, hamming_distance_threshold=t)
    # print('线性搜索耗时: ', time.time() - s)
    # print('线性搜索结果: ', list(results.keys()))

    # s = time.time()
    # results, merkle_tree = mih.query(anchor, hamming_distance_threshold=t)
    # # print(json.dumps(results, indent=4))
    # # print(merkle_tree)
    # print('MIH 搜索耗时: ', time.time() - s)
    # print('MIH 搜索结果: ', list(results.keys()))

    # s = time.time()
    # print('验证结果: ', verify_proof(results, merkle_tree, cham_hash, pk))
    # print('验证耗时: ', time.time() - s)

    """ 3. test add 1000 """
    # with open('add1000.bin', 'rb') as f:
    #     data = f.read()
    # print('待添加的 1000 个 anchor: ', len(data) // 16)
    # s = time.time()
    # for i in range(len(data) // 16):
    #     new_element = data[i*16:(i+1)*16]
    #     print('添加第 {} 个 anchor: {}'.format(i, new_element.hex()))
    #     mih.add(new_element, cham_hash, pk, sk)
    # print('添加 1000 个元素耗时: ', time.time() - s)
    # mih.save()
    
    # print("汉明距离阈值： ", t)
    # s = time.time()
    # results = mih.linear_search(anchor, hamming_distance_threshold=t)
    # print('线性搜索耗时: ', time.time() - s)
    # print('线性搜索结果: ', list(results.keys()))

    # s = time.time()
    # results, merkle_tree = mih.query(anchor, hamming_distance_threshold=t)
    # # print(json.dumps(results, indent=4))
    # # print(merkle_tree)
    # print('MIH 搜索耗时: ', time.time() - s)
    # print('MIH 搜索结果: ', list(results.keys()))

    # s = time.time()
    # print('验证结果: ', verify_proof(results, merkle_tree, cham_hash, pk))
    # print('验证耗时: ', time.time() - s)