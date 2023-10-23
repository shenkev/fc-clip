import os
import json
from detectron2.utils.logger import setup_logger
from contextlib import ExitStack
# import some common libraries
import numpy as np
import cv2
import torch
import itertools
# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, random_color
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config


# import FCCLIP project
from fcclip import add_maskformer2_config, add_fcclip_config
from demo.predictor import DefaultPredictor, OpenVocabVisualizer
from PIL import Image
import imutils
import json

setup_logger()
logger = setup_logger(name="fcclip")

cfg = get_cfg()
cfg.MODEL.DEVICE='cuda'
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_fcclip_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml")
os.system("gdown 1-91PIns86vyNaL3CzMmDD39zKGnPMtvj")
cfg.MODEL.WEIGHTS = './fcclip_cocopan.pth'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)

# def inference(img):
#     im = cv2.imread(img)
#     #im = imutils.resize(im, width=512)
#     outputs = predictor(im)
#     v = OpenVocabVisualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#     panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
#     return Image.fromarray(np.uint8(panoptic_result)).convert('RGB')
    

title = "FC-CLIP"
description = """Gradio demo for FC-CLIP. To use it, simply upload your image, or click one of the examples to load them. FC-CLIP could perform open vocabulary segmentation, you may input more classes (separate by comma).
The expected format is 'a1,a2;b1,b2', where a1,a2 are synonyms vocabularies for the first class. 
The first word will be displayed as the class name.Read more at the links below."""

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2308.02487' target='_blank'>FC-CLIP</a> | <a href='https://github.com/bytedance/fc-clip' target='_blank'>Github Repo</a></p>"

examples = [
    [
        "demo/examples/ADE_val_00000001.jpg",
        "",
        ["ADE (150 categories)", "COCO (133 categories)", "LVIS (1203 categories)"],
    ],
]


coco_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic_with_sem_seg")
ade20k_metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
lvis_classes = open("./fcclip/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
lvis_classes = [x[x.find(':')+1:] for x in lvis_classes]
lvis_colors = list(
    itertools.islice(itertools.cycle(coco_metadata.stuff_colors), len(lvis_classes))
)
# rerrange to thing_classes, stuff_classes
coco_thing_classes = coco_metadata.thing_classes
coco_stuff_classes = [x for x in coco_metadata.stuff_classes if x not in coco_thing_classes]
coco_thing_colors = coco_metadata.thing_colors
coco_stuff_colors = [x for x in coco_metadata.stuff_colors if x not in coco_thing_colors]
ade20k_thing_classes = ade20k_metadata.thing_classes
ade20k_stuff_classes = [x for x in ade20k_metadata.stuff_classes if x not in ade20k_thing_classes]
ade20k_thing_colors = ade20k_metadata.thing_colors
ade20k_stuff_colors = [x for x in ade20k_metadata.stuff_colors if x not in ade20k_thing_colors]

def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append(words)
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]
    print("extra_classes:", extra_classes)
    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if any("COCO" in label for label in label_list):
        demo_thing_classes += coco_thing_classes
        demo_stuff_classes += coco_stuff_classes
        demo_thing_colors += coco_thing_colors
        demo_stuff_colors += coco_stuff_colors
    if any("ADE" in label for label in label_list):
        demo_thing_classes += ade20k_thing_classes
        demo_stuff_classes += ade20k_stuff_classes
        demo_thing_colors += ade20k_thing_colors
        demo_stuff_colors += ade20k_stuff_colors
    if any("LVIS" in label for label in label_list):
        demo_thing_classes += lvis_classes
        demo_thing_colors += lvis_colors

    MetadataCatalog.pop("fcclip_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("fcclip_demo_metadata")
    demo_metadata.thing_classes = demo_thing_classes
    demo_metadata.stuff_classes = demo_thing_classes+demo_stuff_classes
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


output_path = "/home/tshen/projects/fc-clip/out/rsivl"
input_path = "/ptmp/tshen/shared/RSIVL/images"

def inference(image_path, vocab, label_list):
    
    fname = os.path.basename(image_path)
    print("Running {}".format(fname))
    logger.info("building class names")
    vocab = vocab.replace(", ", ",").replace("; ", ";")
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    predictor.set_metadata(demo_metadata)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = OpenVocabVisualizer(im[:, :, ::-1], demo_metadata, instance_mode=ColorMode.IMAGE)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    result = Image.fromarray(np.uint8(panoptic_result)).convert('RGB')
    result.save("{}/{}.jpg".format(output_path, fname))

    segment_labels = outputs["panoptic_seg"][1]
    
    thing_segments = [x for x in segment_labels if x['isthing']]

    for x in thing_segments:
        x['category'] = demo_metadata.thing_classes[x['category_id']]

    stuff_segments = [x for x in segment_labels if not x['isthing']]

    for x in stuff_segments:
        x['category'] = demo_metadata.stuff_classes[x['category_id']]

    with open('{}/{}.json'.format(output_path, fname), 'w') as f:
        json.dump(thing_segments + stuff_segments, f)


import glob

for fpath in glob.glob("{}/*".format(input_path)):
    inference(fpath, "", ["ADE (150 categories)", "COCO (133 categories)", "LVIS (1203 categories)"])
