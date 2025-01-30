import argparse
import torch
from nnUNet22.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    parser = argparse.ArgumentParser(description='Image segmentation')
    # Arguments with defaults
    parser.add_argument('-i', '--input', type=str, 
                       default=r'C:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\machine_learning\LiverTumorSegmentation\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train\imagesTr\im3.nii.gz',
                       help='input image filename (NIfTI) (default: ./MRIImages/input.nii.gz)')
    parser.add_argument('-o', '--output', type=str, 
                       default='./MRIImages/output.nii.gz',
                       help='output image filename (NIfTI) (default: ./MRIImages/output.nii.gz)')
    parser.add_argument('-s', '--state_dict', type=str,
                       default=r'C:\Users\ngocd\OneDrive\Documents\Khiet\bourgogne\machine_learning\1_TotalSegmentor-nnUnet-S_docker\atlas-docker-1.0.1\STUNetTrainer_base_ft__nnUNetPlans__3d_fullres',
                       help='model state_dict folder path to load (default: STUNetTrainer_large_ft__nnUNetPlans__3d_fullres)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='enable verbose output')
    parser.add_argument('-f', '--fold', type=int, default=0,
                       help='fold to use for prediction (default: 0)')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoint_best.pth',
                       help='checkpoint name (default: checkpoint_best.pth)')
    args = parser.parse_args()

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # device=torch.device('cpu'),
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=True
    )

    # Initialize from model folder using proper prototype
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=args.state_dict,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint
    )

    # Predict using proper prototype format
    print(f'Predicting {args.input} to {args.output}')
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[args.input]],
        output_folder_or_list_of_truncated_output_files=[args.output],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

if __name__ == '__main__':
    main()