#!/bin/bash

root=/home/share/huadjyin/home/yinpeng/dataset/sukui/gene_structure/EvalModels/Tiberius
cd $root

ckpt=$root/ckpts/tiberius_weights
learnMSA=$root/learnMSA
fasta_path=/home/share/huadjyin/home/yinpeng/dataset/sukui/gene_structure/EvalDatasets/Homo_Sapiens/GCF_000001405.40_GRCh38.p14_genomic.fna
save_path=$root/outputs/tiberius.hmm_homo

if [ ! -d save_path  ];then
  mkdir -p $save_path
else
  echo $save_path exist
fi

echo $fasta_path
echo $save_path
start_time=$(date +%s.%N)

python bin/tiberius.py \
    --learnMSA $learnMSA \
    --model $ckpt \
    --genome $fasta_path \
    --out $save_path/tiberius.hmm_homo.500k.full_0506.gtf


stage1_end=$(date +%s.%N)
stage1_time=$(echo "$stage1_end - $start_time" | bc)
echo "耗时: $stage1_time 秒"
