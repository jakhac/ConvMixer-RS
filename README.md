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
- [ ] Decrease patch-size coupled with dilated kernels
- [ ] Mosaic augmentation
- [ ] Weight Decay
- [x] Increase RAM too speed up training
- [ ] Train a ResNet model from scratch
- [ ] Log AP/F1 per class in tensorboard
- [ ] Long runs with finetuned AdamW and low learning-rate (test every 5 epochs!)
- [ ] Try different combinations of residual connections
- [x] aug=2 but without constant padding (instead of reflect_padding mode)
- [ ] Investigate wether deeper ConvMixer make up for larger patch_sizes (as hypothesized in paper)

Questions