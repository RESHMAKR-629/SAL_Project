#!/bin/bash

# --- MODIFICATION START: Accept Experiment Name Argument ---
# Usage: sbatch script.sh [experiment_name]
# Example: sbatch script.sh Original
#experiment_name=${1:-default_experiment}
experiment_name="syllable_fine"
echo "Starting Experiment: $experiment_name"
# --- MODIFICATION END ---

. ./cmd.sh
. ./path.sh

# List of speakers
speakers="F01 F03 F04 M01 M02 M03 M04 M05"

# --- PART 1: GLOBAL DATA PREPARATION ---
echo "------------------------------------------------"
echo "Stage 1: Preparing Directory and Dictionary"
echo "------------------------------------------------"

# 1. Run Data Prep (Sorting & spk2utt generation)
local/prepare_data.sh data/all_data

# 2. Extract Features
if [ ! -f data/all_data/feats.scp ]; then
    echo "Extracting MFCCs..."
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/all_data exp/make_mfcc/all_data exp/mfcc
    steps/compute_cmvn_stats.sh data/all_data exp/make_mfcc/all_data exp/mfcc
fi

# 3. Prepare Dictionary & Lang (Using new script)
local/prepare_dict.sh
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

# 4. Prepare LM (Using new script)
local/prepare_lm.sh

# --- PART 2: LOSO LOOP ---
for test_spk in $speakers; do
    echo "=================================================="
    echo "Running LOSO for Test Speaker: $test_spk"
    echo "=================================================="

    train_dir=data/train_$test_spk
    test_dir=data/test_$test_spk
    
    # --- MODIFICATION: Update Experiment Directory ---
    # Old: exp_dir=exp/$test_spk
    exp_dir=exp/$experiment_name/$test_spk
    # -------------------------------------------------
    
    pred_dir=$exp_dir/chain/predictions
    mkdir -p $pred_dir

    # A. Split Data
    # NOTE: Using utils/subset_data_dir.sh handles the spk2utt/utt2spk automatically
    utils/subset_data_dir.sh --spk-list <(echo $test_spk) data/all_data $test_dir
    utils/subset_data_dir.sh --spk-list <(comm -23 <(cut -d' ' -f1 data/all_data/spk2gender | sort) <(echo $test_spk | sort)) data/all_data $train_dir

    # Check if split worked
    if [ ! -f $train_dir/feats.scp ]; then
        echo "Error: Train split failed. feats.scp missing."
        exit 1
    fi

    # B. GMM-HMM Training (Bootstrap)
    if [ ! -f $exp_dir/tri3b/final.mdl ]; then
        echo "Starting GMM-HMM training..."
        steps/train_mono.sh --nj 4 --cmd "$train_cmd" $train_dir data/lang $exp_dir/mono
        steps/align_si.sh --nj 4 --cmd "$train_cmd" $train_dir data/lang $exp_dir/mono $exp_dir/mono_ali
        
        steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 $train_dir data/lang $exp_dir/mono_ali $exp_dir/tri1
        steps/align_si.sh --nj 4 --cmd "$train_cmd" $train_dir data/lang $exp_dir/tri1 $exp_dir/tri1_ali
        
        steps/train_deltas.sh --cmd "$train_cmd" 2500 15000 $train_dir data/lang $exp_dir/tri1_ali $exp_dir/tri2
        steps/align_si.sh --nj 4 --cmd "$train_cmd" $train_dir data/lang $exp_dir/tri2 $exp_dir/tri2_ali
        
        steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 15000 $train_dir data/lang $exp_dir/tri2_ali $exp_dir/tri3b
    fi

    # C. LF-MMI (Chain) Training
    # Only run if not already done
    if [ ! -f $exp_dir/chain/tdnn/final.mdl ]; then
        # Speed Perturbation
        utils/data/perturb_data_dir_speed.sh 0.9 $train_dir ${train_dir}_sp0.9
        utils/data/perturb_data_dir_speed.sh 1.1 $train_dir ${train_dir}_sp1.1
        utils/combine_data.sh ${train_dir}_sp $train_dir ${train_dir}_sp0.9 ${train_dir}_sp1.1
        
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 ${train_dir}_sp $exp_dir/make_mfcc/train_sp $exp_dir/mfcc
        steps/compute_cmvn_stats.sh ${train_dir}_sp $exp_dir/make_mfcc/train_sp $exp_dir/mfcc
        steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" ${train_dir}_sp data/lang $exp_dir/tri3b $exp_dir/tri3b_ali_sp
        
        # Train Chain
        local/chain/run_tdnn.sh ${train_dir}_sp data/lang $exp_dir/tri3b_ali_sp $exp_dir/chain/tree $exp_dir/chain/tdnn
    fi

    # D. Decoding & Scoring
    if [ ! -f $exp_dir/chain/tdnn/decode_test/scoring/10.txt ]; then
        utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $exp_dir/chain/tdnn $exp_dir/chain/tdnn/graph
        
        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 1 --cmd "$decode_cmd" \
          $exp_dir/chain/tdnn/graph $test_dir $exp_dir/chain/tdnn/decode_test
          
        local/score.sh --cmd "$decode_cmd" $test_dir $exp_dir/chain/tdnn/graph $exp_dir/chain/tdnn/decode_test
    fi

    # E. Python CSV Generation
    hyp_file=$exp_dir/chain/tdnn/decode_test/scoring/10.txt
    if [ -f $hyp_file ]; then
        echo "Generating CSV predictions for $test_spk..."
        python3 local/score_predictions.py \
            --ref $test_dir/text \
            --hyp $hyp_file \
            --output_dir $pred_dir \
            --speaker $test_spk
    else
        echo "Error: Hypothesis file $hyp_file not found. Decoding might have failed."
    fi
done
