# Changelog

## (v1) Baseline for depth-emphasized ConvMixer (512-16-9-7)

_h_ and _d_ selected from smaller, but still well performing, model tested in ConvMixer paper

- Ranger21: performs best
- Adam + AdamW: perfom equally good, bump in later epochs (-> tune hparams of AdamW)
- Lamb: good trend but slow, needs way more epochs

## (v2) Baseline for hiddendim-emphasized ConvMixer (960-8-9-7)

_d_ is half of smaller 80%-model from ConvMixer paper, _h_ is largest GPU compatible number

- Ranger21: performs best on test set (might benefit from 30 epochs)
- Adam + AdamW: almost identical results, early overfit after, 20 epochs sufficient
- Lamb: good tendecy but likely needs more epochs, approx. 40

## (v3) Add data augmentation

- v2 with moderate augmentation performs best, results in ranking aug=2 approx aug=1 > aug=3
- reflect vs. constant padding (aug=2 vs aug=4) shows no difference

=> TODO new augmentation directions?

## (v4) Decrease patch sizes

- When decreasing patch size (and increasing internal resolution) the model overfits quickly (around 8 epochs) but has substantially better start
- When lowering patch sizes, hiddendim-based ConvMixers still beat depth-based ConvMixers (like in v1 vs v2)

=> Retrain hiddemdim model with path_sizes 6, 5 and lower LR by 1e-1 (see v5)

## (v5) Patch size with lower LR

- RUNNING

=> Low training duration caused by Ranger21 and smaller patch_size, can we
increase training speed by finetuned AdamW (or dilated kernels)?

## (v6) Vary receptive field, kernel size
Hypothesis: Larger kernel-sizes perform better due to bigger receptive field
Setup: Take currently best working model (v3 with aug=2) and try k_sizes=7/11

- RUNNING

## (v7) Finetune AdamW

- 

