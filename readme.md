1. Install with https://github.com/bytedance/fc-clip/blob/main/INSTALL.md specifically instructions `Example conda environment setup`
2. Download the real demo file https://huggingface.co/spaces/fun-research/FC-CLIP/blob/main/app.py and put it into the root folder ./fc-clip
3. Their demo/demo.py isn't the real demo, it's running a baseline model with different backbones
4. Remove all lines with os.system and gradio from app.py, also remove everything after the inference() function
5. Change cfg.MODEL.DEVICE='cpu' to cfg.MODEL.DEVICE='cuda'
6. Add a __init__.py to /demo/
7. Change `return Image.fromarray(np.uint8(panoptic_result)).convert('RGB')` to first saving the output as `result` and then `result.save(outpath)`
8. Download the model checkpoint https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view and put it into the root folder ./fc-clip
9. Run the inference function like so: inference("/home/tshen/projects/fc-clip/0.jpg", "", ["ADE (150 categories)", "COCO (133 categories)", "LVIS (1203 categories)"])
10. Save the labels using the following,

```    segment_labels = outputs["panoptic_seg"][1]

    thing_segments = [x for x in segment_labels if x['isthing']]

    for x in thing_segments:
        x['category'] = demo_metadata.thing_classes[x['category_id']]

    stuff_segments = [x for x in segment_labels if not x['isthing']]

    for x in stuff_segments:
        x['category'] = demo_metadata.stuff_classes[x['category_id']]

    with open('/home/tshen/projects/fc-clip/out/0.json', 'w') as f:
        json.dump(thing_segments + stuff_segments, f)
```
