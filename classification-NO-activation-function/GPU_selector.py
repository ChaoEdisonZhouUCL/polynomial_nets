"""
Helper functions for PyTorch.

By Zhaoyan@UCL
"""
import re
import os
import time
from datetime import datetime
from socket import gethostname

import numpy as np
from tqdm import tqdm

import torch


def get_gpu_status(n_iter=10, if_print=True):
    # debug: get GPU info
    # with os.popen('nvidia-smi -q -d memory', 'r') as stdout:
    #     info = stdout.read()
    # num_gpu = int(re.findall(r'^[\W]*Attached GPUs[^\n]*', info, re.MULTILINE)[0].split()[-1])
    # mem_total_list = [i.split()[2] for i in re.findall(r'^[\W]*Total[^\n]*', info, re.MULTILINE)]
    # mem_free_list = [i.split()[2] for i in re.findall(r'^[\W]*Free[^\n]*', info, re.MULTILINE)]
    # mem_used_list = [i.split()[2] for i in re.findall(r'^[\W]*Used[^\n]*', info, re.MULTILINE)]

    n_gpu = torch.cuda.device_count()
    gpu_usage_list = np.zeros(n_gpu)
    mem_usage_list = np.zeros(n_gpu)
    if if_print:
        tq = tqdm(total=n_iter, desc='Reading GPU status', unit='it', dynamic_ncols=True, ascii='=>')
    else:
        tq = None
    for _ in range(n_iter):
        with os.popen('nvidia-smi -q -d utilization', 'r') as stdout:
            info_gpu = stdout.read()
        with os.popen('nvidia-smi -q -d memory', 'r') as stdout:
            info_mem = stdout.read()

        temp_mem_list = re.findall(r'FB Memory Usage([\s\w:\d\n]*)\s*BAR1', info_mem)
        temp_mem_total_list = [float(re.findall(r'Total\s*:\s*(\d*)', i)[0]) for i in temp_mem_list]
        temp_mem_used_list = [float(re.findall(r'Used\s*:\s*(\d*)', i)[0]) for i in temp_mem_list]
        mem_usage_list += np.asarray([i/j*100 for i, j in zip(temp_mem_used_list, temp_mem_total_list)])
        gpu_usage_list += np.array([float(i.split(':')[1]) for i in re.findall(r'\s*Gpu\s*:\s[^%]*', info_gpu)])
        # debug:
        # print(f'Gpu usage list (%): {gpu_usage_list}')
        # print(f'Mem usage list (%): {mem_usage_list}')
        time.sleep(1)
        if if_print:
            tq.update(1)
    return n_gpu, gpu_usage_list / n_iter, mem_usage_list / n_iter

def auto_select_GPU(n_iter=10, candidate_list=None, hostname=None, wait=True):
    """Select GPU automatically.

    :param n_iter: (optional, int) Read the GPU status n_iter times. Each iteration will take 1 second.
    :param candidate_list: (list of int). If specified, the function will select GPU with index in the list (start from 0)
                            only. This is to solve the NVIDIA driver bug when there are >8 GPUs.
    :param hostname: (optional, str) The name of the host.
                                        You can easily get it via:
                                            from socket import gethostname
                                            gethostname()
    :param wait: (optional, bool) when the gpus are busy whether do we wait for the free gpu.
    :return: Torch.device object
    """
    init_candidate_list = candidate_list

    while True:
        candidate_list = init_candidate_list
        n_gpu, gpu_usage_list, mem_usage_list = get_gpu_status(n_iter, if_print=(not wait))

        if n_gpu == 0:
            exit(f'{datetime.now()} E No GPU available on this server.')

        if hostname is None:
            pass
        else:
            # Solving following issue
            # https://forums.developer.nvidia.com/t/cuda-peer-resources-error-when-running-on-more-than-8-k80s-aws-p2-16xlarge/45351
            if hostname.split('.')[0] == 'london': # this server has too many gpus.
                candidate_list = [0, 1, 2, 3, 4, 5, 6, 7]

        if candidate_list is None:
            candidate_list = list(range(n_gpu))

        # remove busy gpus from the candidate list.
        candidate_list_new = []
        for i_gpu in candidate_list:
            if gpu_usage_list[i_gpu] >= 95:
                continue
            if mem_usage_list[i_gpu] >= 90:
                continue
            candidate_list_new += [i_gpu]
        # now it is a short list for candidate gpus.
        candidate_list = candidate_list_new
        if len(candidate_list) == 0:
            if wait:
                print(f'{datetime.now()} W All GPUs are too busy on this server. Waiting for free GPU (retry in 60s)...')
                time.sleep(60)
            else:
                exit(f'{datetime.now()} E All GPUs are too busy on this server. Try other servers. ')
        else:
            gpu_usage_list_short = [gpu_usage_list[i] for i in candidate_list]
            mem_usage_list_short = [mem_usage_list[i] for i in candidate_list]

            selected_idx = np.argmin(mem_usage_list_short)
            print(f'{datetime.now()} I Selected GPU: {candidate_list[selected_idx]}; \n')
            print('='*60)
            print(f'\tServer name: {gethostname()}\n'
                  f'\tGpu usage: {gpu_usage_list_short[selected_idx]:.2f}%;'
                  f'\tMem usage: {mem_usage_list_short[selected_idx]:.2f}%.')
            print('-'*60)
            for idx, (i_gpu, i_mem) in enumerate(zip(gpu_usage_list, mem_usage_list)):
                if idx == selected_idx:
                    print(f'\t->\tGPU {idx}:\tGpu usage: {i_gpu:.2f}%;\tMem usage {i_mem:.2f}%')
                else:
                    print(f'\t\tGPU {idx}:\tGpu usage: {i_gpu:.2f}%;\tMem usage {i_mem:.2f}%')
            print('='*60)
            device = torch.device(f'cuda:{candidate_list[selected_idx]}')
            return device


if __name__ == '__main__':
    auto_select_GPU()
