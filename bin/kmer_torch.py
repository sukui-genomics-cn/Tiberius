import torch


def make_k_mers(sequences, k, pivot_left=True):
    """
    Maps one-hot encoded nucleotide sequences to a k-mer representation.
    Args:
        sequences: A tensor of shape (b, L, 5) representing one-hot encoded sequences of length L.
                   Last dimension: [A, C, G, T, N].
        k: Integer, length of the k-mer.
        pivot_left: Boolean, whether to pivot the k-mer to the left or right.
    Returns:
        A tensor of shape (b, L, 4**(k-1), 4). If pivot_left is True, the last dimension corresponds
        to the 4 possible nucleotides in the leftmost position of the k-mer; otherwise, the rightmost.
        If the k-mer contains 'N', it’s expressed equiprobably among the 4 nucleotides.
    """
    b, L, _ = sequences.shape
    n = sequences.shape[-1] - 1  # Alphabet size excluding 'N' (e.g., 4 for ACGT)
    # n = sequences.dtype.type(n)  # Match dtype of input tensor

    # Handle 'N' by distributing probability uniformly over A, C, G, T
    sequences_no_N = sequences[..., :-1]  # [b, L, 4]
    N_pos = (sequences[..., -1:] == 1).to(sequences.dtype)  # [b, L, 1]
    sequences_no_N = sequences_no_N + (1.0 / n) * N_pos  # Equiprobable for 'N'

    # Padding for k-mers that go beyond sequence boundaries
    pad = torch.ones_like(sequences_no_N[:, :k - 1, :], dtype=sequences.dtype) / n  # [b, k-1, 4]

    if pivot_left:
        # Pad at the end
        sequences_padded_no_N = torch.cat([sequences_no_N, pad], dim=-2)  # [b, L+k-1, 4]
        k_mers = sequences_padded_no_N[:, :L, None, :]  # [b, L, 1, 4]
    else:
        # Pad at the start
        sequences_padded_no_N = torch.cat([pad, sequences_no_N], dim=-2)  # [b, L+k-1, 4]
        k_mers = sequences_padded_no_N[:, k - 1:L + k - 1, None, :]  # [b, L, 1, 4]

    # Build k-mers by shifting and multiplying
    range_iter = range(1, k) if pivot_left else range(k - 2, -1, -1)
    for i in range_iter:
        shift_i = sequences_padded_no_N[:, i:L + i, None, :, None]  # [b, L, 1, 4, 1]
        k_mers = k_mers[..., None, :] * shift_i  # Element-wise multiplication
        shape = [b, L] + ([4 ** i, 4] if pivot_left else [4 ** (k - i - 1), 4])
        k_mers = k_mers.reshape(shape)

    return k_mers


def encode_kmer_string(kmer, pivot_left=True, alphabet="ACGT"):
    """
    Converts a k-mer string to a one-hot encoded class representation (i, j).
    Args:
        kmer: String, e.g., "AAT".
        pivot_left: Boolean, whether to pivot the k-mer to the left or right.
        alphabet: String, default "ACGT".
    Returns:
        A tensor of shape (4**(k-1), 4) representing the one-hot encoded k-mer.
        If 'N' is present, it’s equiprobable among A, C, G, T.
    """
    alphabet_with_unknown = alphabet + "N"
    kmer_indices = [alphabet_with_unknown.index(c) for c in kmer]
    kmer_tensor = torch.tensor(kmer_indices)  # [k]

    # One-hot encode the kmer
    one_hot = torch.nn.functional.one_hot(kmer_tensor, num_classes=len(alphabet_with_unknown)).float()  # [k, 5]

    # Use make_k_mers to encode
    encoded_kmers = make_k_mers(one_hot[None, ...], k=len(kmer), pivot_left=pivot_left)  # [1, k, 4**(k-1), 4]

    # Squeeze and return the appropriate slice
    if pivot_left:
        return encoded_kmers.squeeze(0)[0]  # [4**(k-1), 4]
    else:
        return encoded_kmers.squeeze(0)[-1]  # [4**(k-1), 4]


if __name__ == '__main__':
    seqs = torch.randint(0, 4, (2, 10))
    nucleotides = torch.eye(5)[seqs]
    b, l, _ = nucleotides.shape
    left_3mers = make_k_mers(nucleotides, k=3, pivot_left=True)
    left_3mers = torch.reshape(left_3mers, (b, l, 64))

    right_3mers = make_k_mers(nucleotides, k=3, pivot_left=False)
    right_3mers = torch.reshape(right_3mers, (b, l, 64))

    input_3mers = torch.stack([left_3mers, right_3mers], dim=-2)
    print(input_3mers.shape)  # torch.Size([2, 10, 16, 4])


