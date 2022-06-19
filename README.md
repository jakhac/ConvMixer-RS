# ConvMixer-RS
ConvMixer for remote sensing data (BigEarthNet)


TODOs
- [x] Run tests after training
- [x] LR scheduler (implicit in optimizer)
- [x] Predict function (convert preds to labels)
- [x] Optimzer: LAMB, AdamW, Ranger21, SGD optimizer
- [x] Add recommended metrics
- [x] Data Augmentation: Various resizing, cropping
- [x] RandAugment and Mixup _Not possible due to color augmentation_ 
- [x] Decrease patch-size (match interal resolution in paper)
- [x] _cancelled as lower kernels work better_Decrease patch-size coupled with dilated kernels
- [x] Weight Decay
- [x] Increase RAM too speed up training
- [x] Long runs with finetuned AdamW and low learning-rate
- [x] Try different combinations of residual connections
- [x] aug=2 but without constant padding (instead of reflect_padding mode)
- [ ] Train a ResNet model from scratch
- [ ] Do deeper ConvMixer compensate larger psize (hypo to paper)
- [ ] Measure FLOPS or throughput
- [ ] Add Dropout Layers
- [ ] Get Torchmetrics running for 90+ run

Questions