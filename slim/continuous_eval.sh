while true; do python eval_image_classifier.py --checkpoint_path=/Scratch/dm116/train_herox2 --batch_size=32 --model_name=inception_resnet_v2 --dataset_name=herox --dataset_dir=/Scratch/dm116/herox_set/ --preprocessing_name=naturewatch --dataset_split_name=validation >> herox2.log 2>&1; done;

