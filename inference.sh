export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=2 python inference.py --train_config ./configs/cusent.json --test_dir test_files/testsets_youtube --restore_file 900 -t 25

