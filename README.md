# ConvMixer-RS
ConvMixer for remote sensing data (BigEarthNet)


TODOs
- Implement BCELossWithLogits metric as torchmetric class
- Predict function (convert preds to labels)
- Run tests after training


Questions
- SyncBatchNorm?
- Deal with imbalance - weighted sampler?
- Which accuracy for multilabel classification?
- Which regularization techniques?
- Which data augmentations?
- Usage of A100 V100s?