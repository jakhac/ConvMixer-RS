# ConvMixer-RS
ConvMixer for remote sensing data (BigEarthNet)


TODOs
-[x] Run tests after training
-[ ] LR scheduler
-[ ] Predict function (convert preds to labels)
-[ ] Optimzer: LAMB, AdamW, Ranger21, SGD optimizer
-[ ] Add metrics: Recommended metrics
-[ ] Data Augmentation: Various resizing, cropping .. (color-preserving) & mosaic
-[ ] ResNet Transfer Learning 

Questions
- Which loss/metrics? Which accuracy for multilabel classification?
- Deal with imbalance - weighted sampler?
- Which regularization techniques?
- Which data augmentations?