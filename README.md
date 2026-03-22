>📋  A  README.md for code accompanying our paper DiQDiff (WWW'25 Oral)

# Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation


## Training

To train the model on ML-1M:
```
python main.py -- dataset ml-1m -- num_cluster 8 - lambda_intent 0.6 -- lambda_history 1 -- lambda_contra 0.05 -- eval_interval 2 -- patience 10
```

To train the model on Steam:
```
python main.py -- dataset steam -- num_cluster 32 -- lambda_intent 1.2 -- lambda_history 1 -- lambda_contra 0.2 -- eval_interval 2 -- patience 10
```

To train the model on Beauty:
```
python src/main.py -- dataset amazon_beauty -- num_cluster 32 -- lambda_intent=0.4 -- lambda_history=0.6 -- lambda_contra=1 -- eval_interval 2 -- patience 10
```

To train the model on Toys:
```
python src/main.py -- dataset toys -- num_cluster 32 -- lambda_intent 0.4 -- lambda_history 1 -- lambda_contra 1 -- eval_interval 2 -- patience 10
```
