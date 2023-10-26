# Multi Gap Neural Network for Emotion Prediction

idea from the [paper](https://ieeexplore.ieee.org/abstract/document/9191258)

## Network Description
Used the category classification proposed by [Machajdik and Hanbury](https://doi.org/10.1145/1873951.1873965) and the model was trained on dataset proposed by [You et al.(2016)](http://arxiv.org/abs/1605.02677). 

The final categories is divided into 8 categories, and contained `amusement`, `anger`, `awe`, `contentment`, `disgust`, `excitement`, `fear`, and `sadness`. 

It is usually considered that amusement, awe, contentment and excitement are positive emotion, and the others are negative.

The structure are shown below

![MG network architecture](https://github.com/goldeneave/MultiGapEmoPred/blob/4b591ad96b239470b5139a12c7cfc11e2bdea358/image.PNG)

## Train

1. Download the dataset mentioned above
2. Store images in `data` folder group by emotion classes.
2. Split data with `'split_data.py'`
3. Run `object_det.py` and `places_det.py` (Change directory to `training_models` folder)
4. Run `late_fusion2.py` (Change directory to `training_models` folder)


## Use Pretrained Models 
1. Download [FI pretrained weights](https://github.com/goldeneave/MultiGapEmoPred/releases/download/v0.1/weight.zip), Store them in `pretrained_models` folder
2. Store test images in `data/test` folder.
3. Run `late_fusion2.py`

## Preview with Streamlit or Gradio
The code for preview on web also available.

Run on streamlit could use the command `streamlit run demo_str.py`

Run on Gradio could use the command `python demo_gra.py`
