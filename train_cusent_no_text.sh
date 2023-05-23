export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=1 python train.py --train_config configs/cusent_no_text.json
