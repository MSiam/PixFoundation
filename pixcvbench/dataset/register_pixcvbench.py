# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_CVBENCH = {
    "cvbench_ade20k": (
        "validation/images/",
        "validation/FinalSegmentations.json",
    ),
    "cvbench_coco": (
        "coco/images/",
        "FinalSegmentations.json",
    ),

}

def load_cvbench_json(json_file, image_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    def retrieve_annotation(image_id, annotations):
        anns = []
        for ann in annotations:
            if  ann["image_id"] == image_id:
                anns.append(ann)
        return anns

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = {}
    for img_info in json_info["images"]:
        image_id = int(img_info["id"])
        anns = retrieve_annotation(image_id, json_info["annotations"])
        file_name = img_info["file_name"]
        if file_name[0] == '/':
            file_name = file_name[1:]
        image_file = os.path.join(image_dir, file_name)

        if image_id not in ret:
            ret[image_id] = []

        if len(anns) != 0:
            for ann in anns:
                if 'size' not in ann:
                    ann['size'] = Image.open(image_file).size

                ret[image_id].append(
                    {
                        "file_name": image_file,
                        "image_id": image_id,
                        "segments_info": {'segments': ann['segmentation'], 'category': ann['category_id'], 'size': ann['size']}
                    }
                )
        else:
            ret[image_id].append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "segments_info": {'segments': None, 'category': None}
                }
            )

        assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[image_id][0]["file_name"]), ret[image_id][0]["file_name"]
    return ret.items(), json_info["categories"]


def register_NEW(
    name, root, json
):
    DatasetCatalog.register(
        name,
        lambda: load_cvbench_json(json, root),
    )

    MetadataCatalog.get(name).set(
        image_root=root,
        json_file=json,
        ignore_label=255,
        label_divisor=1000,
    )


def register_new_dataset(roots):
    for (
        root, (prefix,
        (predef_root, predef_json)),
    ) in zip(roots, _PREDEFINED_SPLITS_CVBENCH.items()):
        register_NEW(
            prefix,
            os.path.join(root, predef_root),
            os.path.join(root, predef_json),
        )
