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

## (v4) Decrease patch sizes (no aug)

- When decreasing patch size (and increasing internal resolution) the model overfits quickly (around 8 epochs) but has substantially better start
- When lowering patch sizes, hiddendim-based ConvMixers still beat depth-based ConvMixers (like in v1 vs v2)

=> Retrain hiddemdim model with path_sizes 6, 5 and lower LR by 1e-1 (see v5)

## (v5) Patch size with lower LR (no aug)

- Need more epochs (> 40), was improving in all last 5 epochs
- The bump mentioned in v4 is flattened
- About 2-4% worse than best run (v3 with aug=2)

=> Low training speed caused by Ranger21 and smaller patch_size, can we
increase training speed by finetuned AdamW (or dilated kernels)? (see v7)

## (v6) Vary receptive field, kernel size (with aug=2)
Hypothesis: Larger kernel-sizes perform better due to bigger receptive field
Setup: Take currently best working model (v3 with aug=2) and try k_sizes=7/11

- Reject hypothesis?? 7 > 11 in contrast to paper experiments
- TODO what about 5 and 13 - is there a clear trend visible?

## (v7) Finetune AdamW (with aug=2)

-  AdamW is faster and lower LR suffices to cope with quick overfit

## Intermediate results:

_Optimizer_
Ranger21 provides many automated features but is incredibly slow. AdamW with small LR is faster and provides equally good results. 1CyclicLR overfits - small LR is sufficient to avoid loss-bump in latter epochs.
=> Use AdamW with more epochs and small LR (approx. 5e-6 or 1e-5)
or Use Ranger21 with approx 35 epochs

_Augmentation_
Augmentation version 2 works best and intuitively makes sense, too. If needed, more regularization is possible with zero-padding.
=> For now, stay with aug=2 since model and optimizer hparams are still to be tuned

_Kernel-Size_
Surprise, lower kernel sizes work better

_Patch-Size_
Smaller patch-size overfit fastly, however, they theoretically should improve performance. Lower LR shows good potential, but needed more epochs
=> Re-run with LR 1e-5, 50-55 epochs, aug=2


## v8 Combine augmentation, p_size, optimizer findings

AdamW due to training speed
- *aug=2 + p_size=5 + AdamW + LR=1e-5 , epochs=50 + dec=0.05
- aug=2 + p_size=5 + AdamW + LR=5e-6 , epochs=50 + dec=0.05 + ReduceLROnPlateau
- aug=2 + p_size=5 + Ranger21 + LR=1e-5 , epochs=35

=> Combine best run (*) with smaller kernel sizes (see v9)

## v9 based on v8 and v6

- TBD
