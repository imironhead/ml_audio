DATASET_ROOT=/home/ironhead/datasets/tsv_experiments
DATASET_NAME=f1_cr_cmu_us_rms_arctic

EXP_NAME=f1_cr_cmu_us_rms_arctic
EXP_FEATURE_NAME=mel
EXP_FEATURE_BINS=32


DECODE_NAME=f1_cr_cmu_us_rms_arctic
DECODE_DATASET_NAME=f1_cr_cmu_us_rms_arctic
DECODE_FEATURE_NAME=mel
DECODE_FEATURE_BINS=32
DECODE_ROOT=${DATASET_ROOT}/${DECODE_DATASET_NAME}/normal_test/
DECODE_STEP=300000
DECODE_CKPT=./fftnet/ckpt/${DECODE_NAME}_${DECODE_FEATURE_NAME}_${DECODE_FEATURE_BINS}/model.ckpt-${DECODE_STEP}
DECODE_OUTPUT=${DATASET_ROOT}/${DECODE_DATASET_NAME}/fftnet_test_${DECODE_FEATURE_NAME}_${DECODE_FEATURE_BINS}


train:
	pipenv run python -m fftnet.experiment_train \
		--condition_size=${EXP_FEATURE_BINS} \
		--train_dir_path=${DATASET_ROOT}/${DATASET_NAME}/fftnet_train_mel_32/ \
		--ckpt_path=./fftnet/ckpt/${EXP_NAME}/ \
		--logs_path=gs://tsvoice-research-ml-results/fftnet/${EXP_NAME}/ \
		--max_training_step=300000

decode:
	ls ${DECODE_ROOT} | xargs -I % \
		pipenv run python -m fftnet.experiment_decode \
		--ckpt_path=${DECODE_CKPT} \
		--source_wave_path=${DECODE_ROOT}/% \
		--result_wave_path=${DECODE_OUTPUT}/% \
		--feature_name=${DECODE_FEATURE_NAME} \
		--condition_size=${DECODE_FEATURE_BINS} \
		--use_cpu
