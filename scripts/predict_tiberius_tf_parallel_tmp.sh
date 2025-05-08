export CUDA_VISIBLE_DEVICES=1
TIBERIUS_ROOT=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

cd $TIBERIUS_ROOT
#SPECIES_FASTA="/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/test_data/hg38_nop56.fasta"
SPECIES_FASTA="/home/share/huadjyin/home/yinpeng/dataset/sukui/gene_structure/Species/Danio_rerio/GCF_000002035.6_GRCz11_genomic.tmp.fna"
#MODEL_PATH=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/DNA_LLM/outputs/supervised/FishsTiberius7label_Combine_dataset_with_10_fish_tiberius_transformer
MODEL_PATH=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/weights/tiberius_weights

python bin/tiberius.py \
  --genome $SPECIES_FASTA \
  --out /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/danio_rerio_output.tmp.parallel.tf.gtf \
  --model $MODEL_PATH \
  --learnMSA $TIBERIUS_ROOT/learnMSA \
  --batch_size 8
