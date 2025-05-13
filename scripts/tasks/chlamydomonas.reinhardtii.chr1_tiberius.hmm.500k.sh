#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/home/share/huadjyin/home/s_sukui/99_tmp/
root=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius
cd $root

ckpt=$root/weights/tiberius_weights
learnMSA=$root/learnMSA
fasta_path=/home/share/huadjyin/home/yinpeng/dataset/weitong/with_TE/07.chlamydomonas_reinhardtii/07.chlamydomonas_reinhardtii.softmask.genome.fa
save_path=$root/outputs/chlamydomonas.reinhardtii_tiberius.hmm

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
    --chr_prefix "" \
    --out $save_path/chlamydomonas.reinhardtii.chrall_tiberius.hmm.500k_0513.gtf


stage1_end=$(date +%s.%N)
stage1_time=$(echo "$stage1_end - $start_time" | bc)
echo "Cost time: $stage1_time 秒"
