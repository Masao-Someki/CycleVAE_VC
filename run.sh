#!/bin/bash

# stage settings
stage=r123

# speakers to train
all_spks=(jvs001 jvs002)

# directories
wav_dir=wav
figure_dir=figure
data_dir=data
model_dir=model
exp_dir=exp

# wav and extract info
fs=24000
shiftms=5
mcep_dim=49
fftl=1024

# training related
model_name=test_model

# number of cpus
n_jobs=16

. parse_options.sh

#####################################
############# stage -1 ##############

# stage -1
# remove dataset that exists.
if echo ${stage} | grep -q r; then
	rm -r ./log/*
	rm -r ./data/train/*
	rm -r ./data/val/*
	rm -r data/stats/*

fi

#####################################
############# stage 0 ###############
# stage 0
# build histgram of the f0 and npow.
if echo ${stage} | grep -q 0; then
	echo ${all_spks[@]}
	for spk in ${all_spks[@]}; do
		echo Processing ${spk}

		# directory
		mkdir -p ${figure_dir}/${spk}

		# process
		python src/prepro/create_hist.py \
			--n_jobs ${n_jobs} \
			--wav_dir ${wav_dir}/train/${spk} \
			--figure_dir ${figure_dir}/${spk}
	done
fi

#####################################
############ stage 1 ################
# data preprocessing
if echo ${stage} | grep -q 1; then
	# exp directory
	mkdir -p ${exp_dir}/extract

	for spk in ${all_spks[@]}; do
		# train directory
		mkdir -p ${data_dir}/train/${spk}

		# val directory
		mkdir -p ${data_dir}/val/${spk}

		# training data
		python src/prepro/extract_feature.py \
			--log_dir ${exp_dir}/extract \
			--wav_dir ${wav_dir}/train/${spk} \
			--hdf5dir ${data_dir}/train/${spk} \
			--conf_path ./config/speaker/${spk}.conf \
			--fs ${fs} \
			--shiftms ${shiftms} \
			--mcep_dim ${mcep_dim} \
			--fftl ${fftl} \
			--n_jobs ${n_jobs}

		# val data
		python src/prepro/extract_feature.py \
			--log_dir ${exp_dir}/extract \
			--wav_dir ${wav_dir}/val/${spk} \
			--hdf5dir ${data_dir}/val/${spk} \
			--conf_path ./config/speaker/${spk}.conf \
			--fs ${fs} \
			--shiftms ${shiftms} \
			--fs ${fs} \
			--shiftms ${shiftms} \
			--mcep_dim ${mcep_dim} \
			--fftl ${fftl} \
			--n_jobs ${n_jobs}
	done

	# calc total stats
	mkdir -p ${data_dir}/stats

  	python src/prepro/calc_stats.py \
		--hdf5_dir ${data_dir}/train \
    		--stats_dir ${data_dir}/stats
fi

######################################
############ stage 2 ################@
# training
if echo ${stage} | grep -q 2; then
	mkdir -p ${model_dir}
	python src/train.py \
	        --train_dir ${data_dir}/train \
	        --val_dir ${data_dir}/val \
	        --stats_dir ${data_dir}/stats \
	        --total_stats ${data_dir}/total_stats.h5 \
	        --conf_path ./config/vc.conf \
	        --model_dir ${model_dir} \
		--model_name ${model_name} \
	        --decode_dir ${exp_dir}/${model_name} \
	        --log_name ${model_name}
	       	#--resume ${model_dir}/${model_name}.5507.pt
fi

######################################
############ stage 3 ################3
# decode
if echo ${stage} | grep -q 3; then
	decode_dir=${exp_dir}/${model_name}
	mkdir -p ${decode_dir}
	python src/decode.py \
		--test_dir ${data_dir}/val \
		--exp_dir ${decode_dir} \
		--stats_dir ${data_dir}/stats \
		--conf_path ./config/vc.conf \
		--checkpoint ${model_dir}/${model_name}.1448.pt \
		--log_name ${model_name} \
		--fs ${fs} \
		--shiftms ${shiftms} \
		--mcep_dim ${mcep_dim} \
		--fftl ${fftl}
fi

#####################################
############ stage 4 ################
# data preprocessing for vocoder training
if echo ${stage} | grep -q 4; then
        # exp directory
        mkdir -p ${exp_dir}/extract_vocoder

        for spk in ${all_spks[@]}; do
                # training data
                python src/prepro_vocoder/extract_stft.py \
                        --wav_dir ${wav_dir}/train/${spk} \
                        --hdf5dir ${data_dir}/train/${spk} \
                        --fs ${fs} \
                        --fftl ${fftl} \
			--shiftms ${shiftms} \
			--win_size ${fftl} \
                        --n_jobs ${n_jobs}

                # val data
                python src/prepro_vocoder/extract_stft.py \
                        --wav_dir ${wav_dir}/val/${spk} \
                        --hdf5dir ${data_dir}/val/${spk} \
                        --fs ${fs} \
                        --fftl ${fftl} \
			--shiftms ${shiftms} \
			--win_size ${fftl} \
                        --n_jobs ${n_jobs}
        done

	# convert mcep with the trained model.
	python src/prepro_vocoder/convert_mcep.py \
                --data_dir ${data_dir}/train \
                --stats_dir ${data_dir}/stats \
                --conf_path ./config/vc.conf \
                --checkpoint ${model_dir}/${model_name}.1448.pt \
                --log_name ${model_name}

	python src/prepro_vocoder/convert_mcep.py \
                --data_dir ${data_dir}/val \
                --stats_dir ${data_dir}/stats \
                --conf_path ./config/vc.conf \
                --checkpoint ${model_dir}/${model_name}.1448.pt \
                --log_name ${model_name}
fi



