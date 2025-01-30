# Our Proposed nnUnet-based Framework Docker Container
Thanh-Huy Nguyen et.al. (Universite de Bourgogne Europe)

This repository contains a Dockerfile for building a containerized environment to run STU-Net inference on medical images.

## Installation

1. Build the Docker image:
```bash
docker build -t stunet .
```

2. Run the container with mounted data directory:
```bash
docker run -v /path/to/input/data:/app/MRIImages -it --ipc=host stunet
```

## Usage

Run inference inside the container:
```bash
python my_network_infer.py [-h] [-i INPUT] [-o OUTPUT] [-s STATE_DICT] [-v] [-f FOLD] [-c CHECKPOINT]
```

Arguments:
- `-i, --input`: Input image path (default: MRIImages/input.nii.gz)
- `-o, --output`: Output prediction path (default: MRIImages/output.nii.gz) 
- `-s, --state_dict`: Model folder path (default: STUNetTrainer_large_ft__nnUNetPlans__3d_fullres)
- `-f, --fold`: Model fold (default: 0)
- `-c, --checkpoint`: Checkpoint name (default: checkpoint_best.pth)
- `-v, --verbose`: Enable verbose output

Example:
```bash
python my_network_infer.py -i MRIImages/input.nii.gz -o MRIImages/output.nii.gz -v
```

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)

python my_network_infer.py -i ./MRIImages/im0.nii.gz -o ./MRIImages/output.nii.gz -s ./STUNetTrainer_base_ft__nnUNetPlans__3d_fullres -f 0 -c checkpoint_final.pth -v
