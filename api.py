import torch
from nnUNet22.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from flask import Flask, render_template, request
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import base64
import os
import nibabel as nib

app = Flask(__name__)
cmap = ListedColormap(["purple", "green", "yellow"])

def main(input, output, state_dict, verbose, fold, checkpoint):
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        # device=torch.device('cpu'),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )

    # Initialize from model folder using proper prototype
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=state_dict,
        use_folds=(fold,),
        checkpoint_name=checkpoint
    )

    # Predict using proper prototype format
    print(f'Predicting {input} to {output}')
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[input]],
        output_folder_or_list_of_truncated_output_files=[output],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

def nii2plt(np_imgs, cmap="gray"):
    slices = []
    for i in range(np_imgs.shape[-1]):
        np_img = np_imgs[:, :, i]
        fig, ax = plt.subplots()
        if cmap == "gray":
            ax.imshow(np_img, cmap="gray")
        else:
            ax.imshow(np_img, cmap=cmap, vmin=0, vmax=2)
        ax.axis("off")
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png", bbox_inches="tight", pad_inches=0)
        img_bytes.seek(0)
        plt.close(fig)
        slices.append(base64.b64encode(img_bytes.read()).decode("utf-8"))
    return slices

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        gt_file = request.files['gt']
        
        if image_file and gt_file:
            image_file.save(os.path.join('static', 'image.nii.gz'))
            gt_file.save(os.path.join('static', 'gt.nii.gz'))
            
            img_slices = nii2plt(nib.load(os.path.join('static', 'image.nii.gz')).get_fdata())
            gt_slices = nii2plt(nib.load(os.path.join('static', 'gt.nii.gz')).get_fdata(), cmap=cmap)
            
            main(input=os.path.join('static', 'image.nii.gz'), 
                 output=os.path.join('static', 'pred.nii.gz'), 
                 state_dict='STUNetTrainer_base_ft__nnUNetPlans__3d_fullres', 
                 verbose=True, 
                 fold=0, 
                 checkpoint='checkpoint_best.pth')
            
            pred_slices = nii2plt(nib.load(os.path.join('static', 'pred.nii.gz')).get_fdata(), cmap=cmap)
            
            return render_template("index.html", img_slices=img_slices, 
                                   pred_slices=pred_slices, 
                                   gt_slices=gt_slices)
        elif image_file:
            img_slices = nii2plt(nib.load(os.path.join('static', 'image.nii.gz')).get_fdata())
            
            main(input=os.path.join('static', 'image.nii.gz'), 
                 output=os.path.join('static', 'pred.nii.gz'), 
                 state_dict='STUNetTrainer_base_ft__nnUNetPlans__3d_fullres', 
                 verbose=True, 
                 fold=0, 
                 checkpoint='checkpoint_best.pth')
            
            pred_slices = nii2plt(nib.load(os.path.join('static', 'pred.nii.gz')).get_fdata(), cmap=cmap)
            
            return render_template("index.html", img_slices=img_slices, 
                                   pred_slices=pred_slices, 
                                   gt_slices=None)
    return render_template("index.html", img_slices=None, pred_slices=None, gt_slices=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=False)