# Changelog

## (v1) Baseline for depth-emphasized ConvMixer (512-16-9-7)

_h_ and _d_ selected from smaller, but still well performing, model tested in ConvMixer paper

- Ranger21: 
- SGD: 
- Adam + AdamW: 
- Lamb: 

## (v2) Baseline for hiddendim-emphasized ConvMixer (960-8-9-7)

_d_ is half of smaller 80%-model from ConvMixer paper, _h_ is largest GPU compatible number

- Ranger21: performs best on test set (might benefit from 30 epochs)
- SGD: performs worst on test set (needs more epochs .. 45?)
- Adam + AdamW: almost identical results, early overfit after, 20 epochs sufficient
- Lamb: good tendecy but likely needs more epochs, approx. 40

## (v?) Add 3 levels of data augmentation

