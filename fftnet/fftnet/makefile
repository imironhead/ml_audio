GCP_INSTANCE=tsvoice-p100-1
GCP_SSH=gcloud beta compute --project "gifted-audio-230503" ssh --zone "asia-east1-c" "$(GCP_INSTANCE)"
GCP_SCP=gcloud beta compute --project "gifted-audio-230503" scp --zone "asia-east1-c" --recurse


gcp-deploy:
	$(GCP_SSH) -- "cd ~/projects/ml_audio/fftnet/fftnet/; rm -f *.py;"
	tar -czvf package.tgz *.py makefile
	$(GCP_SCP) package.tgz $(GCP_INSTANCE):projects/ml_audio/fftnet/fftnet/
	$(GCP_SSH) -- -t "cd ~/projects/ml_audio/fftnet/fftnet/; tar -xvf package.tgz --directory .;"


PATH_ROOT=/home/ironhead/datasets/tsv_experiments
NAME_DATA=f1_cr_cmu_us_rms_arctic
NAME_TRAIN_NORMAL=normal_train
NAME_TRAIN_EL=el_train
NAME_NPZ=fftnet_train_mel_32


build-dataset:
	pipenv run python dataset.py \
		--source_wav_dir=${PATH_ROOT}/${NAME_DATA}/${NAME_TRAIN_EL} \
		--target_wav_dir=${PATH_ROOT}/${NAME_DATA}/${NAME_TRAIN_NORMAL} \
		--result_npz_dir=${PATH_ROOT}/${NAME_DATA}/${NAME_NPZ} \
		--feature_name=mel \
		--feature_bins=32
