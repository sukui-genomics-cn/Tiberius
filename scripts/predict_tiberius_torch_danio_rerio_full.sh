export CUDA_VISIBLE_DEVICES=0
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT
#SPECIES_FASTA="/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/test_data/hg38_nop56.fasta"
SPECIES_FASTA="/home/share/huadjyin/home/yinpeng/sukui_data/gene_structure/Species/Danio_rerio/GCF_000002035.6_GRCz11_genomic.fna"
python bin/tiberius.py \
  --genome $SPECIES_FASTA \
  --out /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/eval/danio_rerio_output.chr1.torch.prehmm.best.v3.gtf \
  --model_lstm /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNA_LLM/outputs/supervised/FishsTiberius7label_Combine_dataset_with_10_fish_tiberius_transformer/checkpoint-best \
  --learnMSA /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/ \
  --batch_size 4 \
  --seq_len 9999 \
  --torch_model \
  --parallel_factor 1