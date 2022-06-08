# ConvMixer-RS
ConvMixer for remote sensing data (BigEarthNet)


TODOs
- [x] Run tests after training
- [x] LR scheduler
- [ ] Predict function (convert preds to labels)
- [x] Optimzer: LAMB, AdamW, Ranger21, SGD optimizer
- [x] Add metrics: Recommended metrics
- [ ] Data Augmentation: Various resizing, cropping .. (color-preserving) & mosaic
- [ ] RandAugment and Mixup [Reg+Aug vs Datasize](https://arxiv.org/pdf/2106.10270.pdf)
- [ ] Specify precision for tensors
- [ ] Cosine LR scheduling
- [ ] 

Questions
- [ ] Log AP/F1 per class in tensorboard
- [ ] ResNet Transfer Learning / from scratch?