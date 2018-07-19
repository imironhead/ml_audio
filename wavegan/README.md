# Replicated "Synthesizing Audio with Generative Adversarial Networks"

## Train

```
python -m wavegan.experiment_train \
    --train_dir_path=/path/to/training/data/ \
    --ckpt_path=./ckpt/ \
    --log_path=./log/ \
    --gradient_penalty_lambda=1.0 \
    --seed_size=100 \
    --model_size=64 \
    --batch_size=32 \
    --num_channels=1 \
    --num_source_samples=15000 \
    --num_target_samples=16384 \
    --shuffle_phase=False \
    --use_speech_commands=True \
    --use_sc09
```

* train_dir_path: /path/to/training/data/
* ckpt_path: /path/to/checkpoint
* log_path: /path/to/log/
* gradient_penalty_lambda: lambda for gradient penalty of wgan
* seed_size: size of random inputs
* model_size: d in arXiv:1802.04208v1, table 3
* batch_size: size of training batch
* num_channels: number of channels of the training data, should be 1
* num_source_samples: random crop speech commands to this length
* num_target_samples: pad training data to this size, should be 16384
* shuffle_phase: use shuffle phase
* use_speech_commands: train_dir_path contains folders of speech commands (e.g. zero)
* use_sc09: use subset of speech commands (zero to nine)

## Results (trained on SC09 without phase shuffle)

* [farth](../assets/wavegan_00_farth.wav)
* [vault](../assets/wavegan_01_vault.wav)
* [stri](../assets/wavegan_02_stri.wav)
* [nine](../assets/wavegan_03_nine.wav)
* [funk](../assets/wavegan_04_funk.wav)
* [six](../assets/wavegan_05_six.wav)

## NOTE
[Official implementation](https://github.com/chrisdonahue/wavegan).
