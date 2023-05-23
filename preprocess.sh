export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CUDA_VISIBLE_DEVICES=2 python preprocess.py --config configs/cusent_no_text.json
# CUDA_VISIBLE_DEVICES=2 python preprocess.py --config configs/cusent.json
CUDA_VISIBLE_DEVICES=3 python preprocess.py --config configs/vctk_no_text.json

