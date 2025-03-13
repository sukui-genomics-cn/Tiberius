export CUDA_VISIBLE_DEVICES=0
TIBERIUS_ROOT=/home/sukui/03.project/Tiberius
#DATA_ROOT=/data/sukui_data/01_data/01_genomics_data/gene_structure
export TF_CPP_MIN_LOG_LEVEL=0

cd $TIBERIUS_ROOT
python bin/train_in_multispecies.py \
  --data /home/sukui/02.data/gene_structure/fish_10/ \
  --dataset_name Acanthochromis_polyacanthus \
  --train_species_file train \
  --val_data val \
  --out /home/sukui/03.project/Tiberius/outputs/combine_dataset_with_10_fish \
  --hmm
