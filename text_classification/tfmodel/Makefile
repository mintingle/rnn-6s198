all: train convert

.PHONY train:
train:
	python train.py

.PHONY convert:
convert:
	python checkpoints/checkpoint_dumper.py --model_type=tensorflow --output_dir=dljs -checkpoint_file=tfmodel

