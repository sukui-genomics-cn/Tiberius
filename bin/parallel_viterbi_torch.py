import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd  # 假设返回的是 DataFrame
from torchcrf import CRF


# 定义一个简单的 HMM 层（假设）
from kmer_torch import make_k_mers


# PyTorch 不需要显式的 Input 层，直接定义模型
class PreHMMModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PreHMMModel, self).__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.dense(x))


class FullModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FullModel, self).__init__()
        self.pre_hmm = PreHMMModel(in_dim, out_dim)
        self.hmm_layer = CRF(out_dim)
        self.emision_layer = GeneEmissionModel(out_dim, out_dim)
        self.seq_emission_layer = GeneSeqEmissionModel()

    def forward(self, x):
        emissions = self.pre_hmm(x)
        # 这里可以直接调用 viterbi 或其他逻辑
        return emissions



class GeneEmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(GeneEmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.B = torch.nn.Parameter(torch.randn(N, M))

    def forward(self, x_t):
        log_emission_matrix = torch.nn.functional.log_softmax(self.B, dim=1)
        if x_t.shape[-1] == self.M:
            emit = torch.einsum('...s,qs->...q', x_t, log_emission_matrix)
        else:
            emit = log_emission_matrix[:, x_t].transpose(0, 1)  # change 0 dim to 1 dim

        return emit


class GeneSeqEmissionModel(torch.nn.Module):
    def __init__(self):
        super(GeneSeqEmissionModel, self).__init__()
        codon_probs = np.load('codon_probs.npy') # 64 * 15
        self.codon_probs = torch.from_numpy(codon_probs).float()
        self.nucleotide_probs = torch.nn.Parameter(torch.ones(3, 4) / 4)

    def forward(self, nucleotides):
        batch, length, dim = nucleotides.shape
        assert dim == 5, "nucleotides should have 5 channels, which including AGCTN"
        nucleotides = torch.reshape(nucleotides, [-1, length, 5])
        # compute probabilities to start the first exon or intorns and the probabilities to end the last exon or intorns
        left_3mers = make_k_mers(nucleotides, k=3, pivot_left=True)
        left_3mers = torch.reshape(left_3mers, (batch, length, 64))

        right_3mers = make_k_mers(nucleotides, k=3, pivot_left=False)
        right_3mers = torch.reshape(right_3mers, (batch, length, 64))

        input_3mers = torch.stack([left_3mers, right_3mers], dim=-2)
        codon_emission_probs = torch.einsum("...rs,rqs->...rq", input_3mers, self.codon_probs)
        codon_emission_probs = torch.prod(codon_emission_probs, dim=-2)
        codon_emission_probs = torch.concat(
            [torch.ones_like(codon_emission_probs[..., :6]) / 4096.0, codon_emission_probs], dim=-1)
        codon_emission_probs += 1e-7

        # compute probabilities in nucleotides at exons.
        nucleotides_no_N = nucleotides[..., :4] + nucleotides[..., 4:] / 4
        nucleotide_emission_probs = torch.einsum('...j,kj->...k', nucleotides_no_N, self.nucleotide_probs)

        return codon_emission_probs, nucleotide_emission_probs


# 创建模型
def make_model(in_dim=5, out_dim=15):
    # 创建 pre-HMM 模型
    pre_hmm_model = PreHMMModel(in_dim, out_dim)
    # 创建完整模型（添加 HMM 层）
    model = FullModel(in_dim, out_dim)

    return model, pre_hmm_model


# 生成序列
def gen_seqs(batch_size, seq_len):
    # 生成 i.i.d. 核苷酸序列，不含 N
    ind = torch.randint(0, 4, size=(batch_size, seq_len))
    ind = torch.eye(5)[ind].float()  # One-hot 编码，[A, C, G, T, N]
    labels = torch.randint(0, 15, size=(batch_size, seq_len))  # 生成标签

    return ind, labels  # 转换为 PyTorch 张量


# 运行实验
def run_experiments(batch_size):
    count_compile_time = False  # 是否计算编译时间
    num_reps = 10  # 重复次数
    sqrt_lengths = list(range(2, 10)) + list(range(10, 101, 10))
    print(f"sqrt_lengths: {sqrt_lengths}")

    # 创建模型
    model, pre_hmm_model = make_model()
    hmm_layer = model.hmm_layer  # 直接访问 HMM 层
    print(f"hmm layer: {hmm_layer}")

    # 生成数据
    L = 16
    seqs, labels = gen_seqs(batch_size, L)
    dummy_hmm_input = pre_hmm_model(seqs)  # 前向传播

    # 不计入编译时间（PyTorch 不需要显式编译，但首次运行会稍慢）
    if not count_compile_time:
        print(f"seqs: {seqs.shape}, dummy_hmm_input: {dummy_hmm_input.shape}")
        print(f"seqs[0]: {seqs[0]}, dummy_hmm_input[0]: {dummy_hmm_input[0]}")
        gen_emi = model.emision_layer(dummy_hmm_input)
        codon_emission_probs, nucleotide_emission_probs = model.seq_emission_layer(seqs)
        full_emission = dummy_hmm_input * codon_emission_probs
        nucleotide_emission_probs = torch.concat(
            [torch.ones_like(full_emission[..., :1 + 3]) / 4.,
             nucleotide_emission_probs,
             torch.ones_like(full_emission[..., 1 + 6:]) / 4.],
            dim=-1
        )
        full_emission = full_emission * nucleotide_emission_probs
        viterbi_seqs = hmm_layer(full_emission, labels)

    # 计时实验
    times = []
    start = time.time()
    for _ in range(num_reps):
        viterbi_seqs = hmm_layer.viterbi(dummy_hmm_input, seqs)
    end = time.time()
    times.append((end - start) / num_reps)

    # 返回 DataFrame（假设实验结果需要整理）
    df = pd.DataFrame({
        "sqrt_lengths": [L],  # 这里仅用了一个长度 L，可以扩展
        "time": times
    })
    return df


# 测试
if __name__ == "__main__":
    batch_size = 32
    df = run_experiments(batch_size)
    print(df)
