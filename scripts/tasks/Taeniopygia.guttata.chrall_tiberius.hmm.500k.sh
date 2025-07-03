#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
root=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
cd $root

ckpt=$root/weights/tiberius_weights
learnMSA=$root/learnMSA
fasta_path=/home/share/huadjyin/home/s_sukui/02_data/gene_structure/EvalDatasets/Taeniopygia_guttata/GCF_048771995.1_bTaeGut7.mat_genomic.fna.gz
save_path=$root/outputs/Taeniopygia.guttata_tiberius.hmm

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
    --batch_size 2 \
    --out $save_path/Taeniopygia.guttata.chrall_.tiberius.hmm.500k_0802.gtf


stage1_end=$(date +%s.%N)
stage1_time=$(echo "$stage1_end - $start_time" | bc)
echo "耗时: $stage1_time 秒"
echo "start 耗时: $start_time 秒"
echo "end 耗时: $stage1_end 秒"
