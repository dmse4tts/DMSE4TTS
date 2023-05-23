export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=0 python train.py --train_config configs/cusent.json
