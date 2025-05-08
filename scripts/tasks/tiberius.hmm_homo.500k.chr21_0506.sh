#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

root=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
cd $root

ckpt=$root/weights/tiberius_weights
learnMSA=$root/learnMSA
fasta_path=/home/share/huadjyin/home/yinpeng/dataset/sukui/gene_structure/EvalDatasets/Homo_Sapiens/GCF_000001405.40_GRCh38.p14_genomic.chr21.fna
save_path=$root/outputs/tiberius.hmm_homo

if [ ! -d save_path  ];then
  mkdir -p $save_path
else
  echo $save_path exist
fi

echo "fast: $fasta_path"
echo "output path: $save_path"
start_time=$(date +%s.%N)

python bin/tiberius.py \
    --learnMSA $learnMSA \
    --model $ckpt \
    --genome $fasta_path \
    --out $save_path/tiberius.hmm_homo.500k.chr21_0506.gtf


stage1_end=$(date +%s.%N)
stage1_time=$(echo "$stage1_end - $start_time" | bc)
echo "耗时: $stage1_time 秒"
