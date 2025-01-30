#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
from functools import lru_cache
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnunetv2.paths import nnUNet_raw


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    """Get unique identifiers from dataset folder."""
    files = subfiles(folder, suffix=file_ending, join=False)
    # Extract base filename without extension
    crop = len(file_ending)
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files



def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    print(files)
    for f in identifiers:
        p = re.compile(re.escape(f) + r"" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    """Get filenames of training images and their corresponding targets."""
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        # Initialize the dataset dictionary
        dataset = {}
        # Process each entry in the dataset list
        for item in dataset_json['dataset']:
            # Get image ID from the filename (e.g., 'im0' from 'imagesTr/im0.nii.gz')
            # image_id = os.path.splitext(os.path.basename(item['image']))[0]
            image_id = os.path.basename(item['image']).split('.')[0]
            # Create absolute paths
            image_path = os.path.abspath(join(raw_dataset_folder, item['image'])) if not os.path.isabs(item['image']) else item['image']
            label_path = os.path.abspath(join(raw_dataset_folder, item['label'])) if not os.path.isabs(item['label']) else item['label']
            # Store in dataset dictionary with the expected structure
            dataset[image_id] = {
                'images': [image_path],  # Wrap in list to maintain compatibility
                'label': label_path
            }
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'], identifiers)
        segs = [join(raw_dataset_folder, 'labelsTr', i + dataset_json['file_ending']) for i in identifiers]
        dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    
    return dataset


if __name__ == '__main__':
    print(get_filenames_of_train_images_and_targets(join(nnUNet_raw, 'Dataset002_Heart')))
