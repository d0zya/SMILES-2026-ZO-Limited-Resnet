# Reproducibility Instructions

Environment:
- Python 3.13
- `torch==2.10.0`
- `torchvision==0.25.0`
- `tqdm==4.67.1`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the official evaluation script:

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 256 \
    --n_batches 32 \
    --output results.json
```

Important implementation details:
- The final solution is self-contained and works with the provided evaluation script.
- The final optimizer updates only `fc.bias`.
- The final head is initialized with a data-dependent prototype-based initialization built from CIFAR100 train features extracted by the pretrained ResNet18 backbone.

# Final Solution Description

Final approach:
- The main improvement came from replacing random head initialization with a smarter initialization based on `weight imprinting`.
- Instead of starting the new CIFAR100 head from random weights, I build one prototype vector per class using features from the pretrained ImageNet ResNet18 backbone.
- For each CIFAR100 class, I take train images, extract backbone features, compute a class prototype, and use that prototype as the corresponding row of `fc.weight`.
- `fc.bias` is initialized to zero.

Why this helped most:
- Random initializations such as Kaiming/Xavier produced a very weak initialized head, around `0.0121` top-1 accuracy.
- The prototype-based initialization immediately produced a strong initialized classifier, around `0.5212` top-1 accuracy.
- In practice, this initialization was much more important than the pseudo-gradient fine-tuning itself.

Training-time data choices:
- I used a class-balanced training subset in `train_data.py`.
- I kept moderate augmentation only.
- I removed `AutoAugment` and `RandomErasing`; without them, the metric improved by about `0.0002`.
- My interpretation is that stronger augmentation was too noisy for this setup.

Zero-order fine-tuning choice:
- I only optimized `fc.bias`.
- Once `fc.weight` is initialized with class prototypes, updating it with pseudo-gradients tends to destroy the useful structure of the imprinted weights.
- In contrast, `fc.bias` starts from zeros and can be adjusted more safely with a zero-order method.

Final result:
- `val_accuracy_top1_imagenet_head = 0.0037`
- `val_accuracy_top1_init_head = 0.5212`
- `val_accuracy_top1_finetuned = 0.5222`

The main contribution to the final metric was the initialization strategy. The pseudo-gradient stage provided only a very small gain in the best run and often degraded performance.

# Experiments And Failed Attempts

## 1. Random head initialization

I first tried standard random initializations such as Kaiming and Xavier.

Result:
- The initialized head stayed around `0.0121`.
- This was far worse than the prototype-based approach.

Conclusion:
- Standard random initialization was not competitive for this assignment.

## 2. Weight imprinting / prototype initialization

This was the key successful idea.

What I tried:
- Prototype-based head initialization using features from the pretrained ResNet18 backbone.
- I also tested different prototype aggregation strategies.

Conclusion:
- This gave the largest improvement by far and became the final solution.

## 3. Stronger augmentations

What I tried:
- `AutoAugment`
- `RandomErasing`

Result:
- They did not help the final metric.
- Removing them improved the result by about `0.0002`.

Conclusion:
- In this setting, heavier augmentation introduced extra noise and was not useful.

## 4. Updating `fc.weight`

What I tried:
- Fine-tuning both `fc.weight` and `fc.bias` with pseudo-gradients.

Result:
- This usually hurt performance.
- The imprinted classifier head was already strong, and updating the weight matrix tended to corrupt it.

Conclusion:
- The final setup updates only `fc.bias`.

## 5. Pseudo-gradient optimization behavior

I tried multiple zero-order variants and hyperparameter settings, but in this task the pseudo-gradient stage mostly hurt an already strong initialized head.

Observed runs with `layers_tuned = ["fc.bias"]`:

| `n_batches` | `batch_size` | `val_accuracy_top1_init_head` | `val_accuracy_top1_finetuned` |
|---|---:|---:|---:|
| 32 | 256 | 0.5212 | 0.5222 |
| 64 | 128 | 0.5212 | 0.5212 |
| 128 | 64 | 0.5212 | 0.5211 |
| 256 | 32 | 0.5212 | 0.5204 |
| 1028 | 8 | 0.5212 | 0.5197 |
| 2048 | 4 | 0.5212 | 0.5157 |

Conclusion:
- More pseudo-gradient steps were not better.
- In fact, smaller batches and many steps tended to degrade the model more.
- The best result came from a conservative setup: fewer steps, larger batch, and bias-only updates.

## Final takeaway

For this assignment, the best practical strategy was:
- build a strong initialized classifier head with prototype-based weight imprinting;
- keep augmentation moderate;
- avoid updating `fc.weight`;
- use pseudo-gradients only very conservatively on `fc.bias`.

The main lesson from my experiments is that, in this setup, zero-order fine-tuning is much less important than starting from a strong head initialization, and aggressive pseudo-gradient updates mostly damage a good solution.
