export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=2 python inference_no_text.py --train_config ./configs/cusent_no_text.json --test_dir test_files/testsets_youtube/wav --restore_file 900 -t 25

# CUDA_VISIBLE_DEVICES=2 python inference_no_text.py --train_config ./configs/vctk_no_text.json --test_dir test_files/testsets_youtube/wav --restore_file 750 -t 25

