#!/bin/bash
#
# Complete FLAML workflow with CLEAN directory structure
#
# This script runs seven sequential steps for each target:
# 1. FLAML optimization (flaml-1-optimize.py)
# 2. Extract best configurations (flaml-2-extract.py)
# 3. Train models using best configs and generate predictions (flaml-3-train_on_best_configs.py)
# 3A. Generate out-of-fold predictions on training set (flaml-3-make_predictions_on_training_with_best_config.py)
# 4. Create consensus predictions from individual models (flaml-4-consensus.py)
# 5. Semi-supervised learning with unlabeled data (flaml-4-train_semisupervised.py)
# 6. Create final summary and best ensemble submission (flaml-9-summarize.py)
#
# NEW DIRECTORY STRUCTURE:
# {output_base}/
# ├── {target_1}/
# │   ├── flaml_logs/
# │   ├── configs/
# │   ├── predictions/
# │   ├── consensus/
# │   ├── semisupervised/
# │   ├── submissions/
# │   └── results.csv
# ├── {target_2}/ ...
# ├── {target_3}/ ...
# ├── {target_4}/ ...
# └── final_submission.csv
#

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# === DATASET AND DESCRIPTOR CONFIGURATION ===
# Explicitly specify the descriptor set name (must match directory in 4-datasets)
#DESCRIPTOR_NAME='2D+fp_v2+embeddings+dyes_similarity+3D_XTB+3Dspectro'
#DESCRIPTOR_NAME='2D+fp_v2+embeddings+dyes_similarity+3D_XTB+3Dspectro+QM9pred+easyTargetPreds'
# DESCRIPTOR_NAME='2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds'
#DESCRIPTOR_NAME='2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds+xtb3+Leadboard'
DESCRIPTOR_NAME='2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds+xtb3_20260107'

# Base directories
DATA_BASE_DIR="./4-datasets"
OUTPUT_BASE_DIR="./5-predictions"

# Experiment/run identifier (appended to output directory name)
EXPERIMENT_SUFFIX="ensWeights+warmStartFluor480plus_2"  # Can be empty: ""

# Construct output directory name
output_base="${OUTPUT_BASE_DIR}/5-flaml-${DESCRIPTOR_NAME}_${EXPERIMENT_SUFFIX}"

# === TARGET DATASETS ===
# Explicitly list which targets to process
TARGETS=(
    "euos25_challenge_train_fluorescence480plus"
#    "euos25_challenge_train_transmittance450plus"
#    "euos25_challenge_train_transmittance340"
#    "euos25_challenge_train_fluorescence340_450"
)

# === GLOBAL TEST DATASET ===
# Name of the global test dataset directory
GLOBAL_TEST_NAME="euos25_challenge_test"

# === TRAINING CONFIGURATION ===
jobs=1
budget=4520 # 3 dni = 4320 minutes  # Time budget in MINUTES for FLAML optimization
n_best=4     # Number of best methods to train in step 3
USE_WEIGHTS=1  # Enable sample weighting for imbalanced data (0=disabled, 1=enabled)
WEIGHT_STRATEGY="ens"  # Weighting strategy: ens (Effective Number of Samples - default), balanced (sklearn), sqrt (square root)

# === SEMI-SUPERVISED LEARNING PARAMETERS ===
semisup_strategy="all"  # Options: simple, iterative, ensemble, cotraining, all
semisup_topk=4              # Number of top models to use for semi-supervised
semisup_confidence=0.95       # Confidence threshold
semisup_agreement=0.95        # Agreement threshold for ensemble
semisup_iterations=30         # Iterations for iterative/cotraining
semisup_samples_per_iter=50  # Samples per iteration

# === CONTROL FLAGS - Which steps to run (set to 0 to skip) ===
RUN_OPTIMIZATION=1
RUN_EXTRACTION=1
RUN_TRAINING=1
RUN_TRAIN_OOF=1        # Generate out-of-fold predictions on training set
RUN_CONSENSUS=1
RUN_SEMISUPERVISED=1
RUN_SUMMARIZE=1

# ============================================================================
# LEGACY COMPATIBILITY (for old code that references these)
# ============================================================================
dir="$DATA_BASE_DIR/"  # Old variable name, kept for compatibility
desc_prefix="$DESCRIPTOR_NAME"  # Old variable name, kept for compatibility

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_step() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

log_substep() {
    echo "--------------------------------------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------------------------------------"
}

log_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_file_exists() {
    local file="$1"
    local description="$2"

    if [ -f "$file" ]; then
        echo "✓ $description exists: $file"
        return 0
    else:
        echo "✗ $description not found: $file"
        return 1
    fi
}

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

dothings(){
    local target_name="$1"
    local target_start_time=$(date +%s)

    log_step "Processing: ${target_name}"
    log_timestamp "START processing target: ${target_name}"

    # Construct explicit paths using configuration variables
    local dataset_dir="${DATA_BASE_DIR}/${target_name}/${DESCRIPTOR_NAME}"
    local train_file="${dataset_dir}/train.csv.gz"
    local test_file="${dataset_dir}/test.csv.gz"
    local global_test_file="${DATA_BASE_DIR}/${GLOBAL_TEST_NAME}/${DESCRIPTOR_NAME}/train.csv.gz"

    # Output directory for this target
    local target_output_dir="${output_base}/${target_name}"

    # Validate that input files exist
    if [ ! -f "$train_file" ]; then
        echo "✗ ERROR: Training file not found: $train_file"
        echo "  Check that DESCRIPTOR_NAME='${DESCRIPTOR_NAME}' is correct"
        echo "  Check that target directory exists: ${DATA_BASE_DIR}/${target_name}/"
        return 1
    fi

    echo "Configuration:"
    echo "  Target: $target_name"
    echo "  Descriptor: $DESCRIPTOR_NAME"
    echo "  Dataset directory: $dataset_dir"
    echo "  Training file: $train_file"
    echo "  Test file: $test_file"
    echo "  Global test file: $global_test_file"
    echo "  Output directory: $target_output_dir"
    echo ""

    # Create organized subdirectories
    mkdir -p "$target_output_dir"
    mkdir -p "${target_output_dir}/flaml_logs"
    mkdir -p "${target_output_dir}/configs"
    mkdir -p "${target_output_dir}/predictions"
    mkdir -p "${target_output_dir}/consensus"
    mkdir -p "${target_output_dir}/semisupervised"
    mkdir -p "${target_output_dir}/submissions"

    # Define file paths using clean structure
    local flaml_log="${target_output_dir}/flaml_logs/optimization.log"
    local best_configs_json="${target_output_dir}/configs/best_configs.json"

    # ============================================================================
    # STEP 1: Run FLAML optimization
    # ============================================================================
    if [ $RUN_OPTIMIZATION -eq 1 ]; then
        log_substep "STEP 1: Running FLAML optimization (time budget: ${budget} minutes)"
        log_timestamp "START Step 1: FLAML optimization"
        local step1_start=$(date +%s)

        if [ ! -f "$flaml_log" ]; then
            echo "Starting FLAML optimization..."

            # Build command with optional --use-weights flag and weight strategy
            cmd="python flaml-1-optimize.py \"$dataset_dir\" --output-dir \"$target_output_dir\" --time-budget \"$budget\""

            if [ $USE_WEIGHTS -eq 1 ]; then
                cmd="$cmd --use-weights --weight-strategy \"$WEIGHT_STRATEGY\""
                echo "  Sample weighting: ENABLED (strategy: $WEIGHT_STRATEGY)"
            else
                echo "  Sample weighting: DISABLED"
            fi

            # Execute the command
            eval $cmd

            if [ $? -eq 0 ]; then
                local step1_end=$(date +%s)
                local step1_duration=$((step1_end - step1_start))
                log_timestamp "FINISH Step 1: FLAML optimization (duration: ${step1_duration}s / $((step1_duration/60))m)"
                echo "✓ FLAML optimization completed successfully"
            else
                log_timestamp "FAILED Step 1: FLAML optimization"
                echo "✗ FLAML optimization failed"
                return 1
            fi
        else
            echo "⊘ FLAML log already exists, skipping optimization"
            echo "  Log file: $flaml_log"
            log_timestamp "SKIP Step 1: FLAML optimization (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 1 (FLAML optimization) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # STEP 2: Extract best configurations
    # ============================================================================
    if [ $RUN_EXTRACTION -eq 1 ]; then
        log_substep "STEP 2: Extracting best configurations from FLAML log"
        log_timestamp "START Step 2: Config extraction"
        local step2_start=$(date +%s)

        if [ ! -f "$best_configs_json" ]; then
            if [ ! -f "$flaml_log" ]; then
                echo "✗ Error: FLAML log file not found: $flaml_log"
                echo "  Cannot extract configs without FLAML log"
                log_timestamp "FAILED Step 2: Config extraction (FLAML log not found)"
                return 1
            fi

            echo "Found FLAML log: $flaml_log"
            echo "Extracting best configurations..."

            python flaml-2-extract.py "$flaml_log" \
                -n 10 \
                -o "${target_output_dir}/configs"

            if [ $? -eq 0 ]; then
                local step2_end=$(date +%s)
                local step2_duration=$((step2_end - step2_start))
                log_timestamp "FINISH Step 2: Config extraction (duration: ${step2_duration}s)"
                echo "✓ Best configurations extracted successfully"
                echo "  Output: $best_configs_json"
            else
                log_timestamp "FAILED Step 2: Config extraction"
                echo "✗ Configuration extraction failed"
                return 1
            fi
        else
            echo "⊘ Best configs already exist, skipping extraction"
            echo "  Config file: $best_configs_json"
            log_timestamp "SKIP Step 2: Config extraction (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 2 (config extraction) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # STEP 3: Train models using best configurations
    # ============================================================================
    if [ $RUN_TRAINING -eq 1 ]; then
        log_substep "STEP 3: Training top ${n_best} models and generating predictions"
        log_timestamp "START Step 3: Model training"
        local step3_start=$(date +%s)

        # Check if submission files already exist
        submission_count=$(find "${target_output_dir}/submissions" -mindepth 1 -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)

        echo "Found $submission_count submission file(s) for ${target_name}"

        if [ "$submission_count" -lt "$n_best" ]; then
            echo "Training models with best configurations..."

            # Paths are already defined at the top of dothings()
            # train_file, test_file, global_test_file

            echo "Checking for global test file..."
            if [ -f "$global_test_file" ]; then
                echo "  ✓ Found global test file: $global_test_file"
            else
                echo "  ✗ Global test file not found: $global_test_file"
                echo "  This is OK - will skip global test predictions"
            fi
            echo ""

            # Build command with explicit file paths
            cmd="python flaml-3-train_on_best_configs.py \"$best_configs_json\" \
                -n \"$n_best\" \
                -o \"$target_output_dir\" \
                --random-state 42 \
                --train-file \"$train_file\" \
                --test-file \"$test_file\""

            # Add global test file if it exists
            if [ -f "$global_test_file" ]; then
                echo "  ✓ Found global test file"
                cmd="$cmd --global-test-file \"$global_test_file\""
            else
                echo "  ✗ Global test file not found at expected location"
                echo "  This is OK - will skip global test predictions"
            fi
            echo ""

            # Execute the command
            eval $cmd

            if [ $? -eq 0 ]; then
                local step3_end=$(date +%s)
                local step3_duration=$((step3_end - step3_start))
                log_timestamp "FINISH Step 3: Model training (duration: ${step3_duration}s / $((step3_duration/60))m)"
                echo "✓ Model training completed successfully"
                echo ""
                echo "Generated files:"
                echo "  - Test results: ${target_output_dir}/results.csv"
                echo "  - Test predictions: ${target_output_dir}/predictions/test_predictions.csv"
                echo "  - Global predictions: ${target_output_dir}/predictions/global_test_predictions.csv"
                echo "  - Submission files: ${target_output_dir}/submissions/*.txt"
            else
                log_timestamp "FAILED Step 3: Model training"
                echo "✗ Model training failed"
                return 1
            fi
        else
            echo "⊘ Submission files already exist (found $submission_count), skipping training"
            log_timestamp "SKIP Step 3: Model training (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 3 (model training) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # STEP 3A: Generate out-of-fold predictions on training set
    # ============================================================================
    if [ $RUN_TRAIN_OOF -eq 1 ]; then
        log_substep "STEP 3A: Generating out-of-fold predictions on training set"
        log_timestamp "START Step 3A: OOF training predictions"
        local step3a_start=$(date +%s)

        # Check if training predictions already exist
        train_pred_file="${target_output_dir}/predictions/train_predictions.csv"

        if [ ! -f "$train_pred_file" ]; then
            if [ ! -f "$best_configs_json" ]; then
                echo "✗ Error: Best configs file not found: $best_configs_json"
                echo "  Cannot generate training predictions without configs"
                log_timestamp "FAILED Step 3A: OOF training predictions (configs not found)"
                return 1
            fi

            echo "Generating out-of-fold predictions for training set..."
            echo "  This will create 5-fold CV predictions for the training data"
            echo "  Method: Stratified K-Fold cross-validation"
            echo "  Output: Predictions in same order as training samples"

            python flaml-3-make_predictions_on_training_with_best_config.py "$best_configs_json" \
                -n "$n_best" \
                --train-file "$train_file" \
                --n-folds 5 \
                --random-state 42

            if [ $? -eq 0 ]; then
                local step3a_end=$(date +%s)
                local step3a_duration=$((step3a_end - step3a_start))
                log_timestamp "FINISH Step 3A: OOF training predictions (duration: ${step3a_duration}s / $((step3a_duration/60))m)"
                echo "✓ Out-of-fold predictions generated successfully"
                echo ""
                echo "Generated files:"
                echo "  - Training predictions: $train_pred_file"
                echo "  - Training report: ${target_output_dir}/predictions/train_predictions.txt"
            else
                log_timestamp "FAILED Step 3A: OOF training predictions"
                echo "✗ Out-of-fold prediction generation failed"
                return 1
            fi
        else
            echo "⊘ Training predictions already exist, skipping"
            echo "  Predictions file: $train_pred_file"
            log_timestamp "SKIP Step 3A: OOF training predictions (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 3A (OOF training predictions) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # STEP 4: Create consensus predictions
    # ============================================================================
    if [ $RUN_CONSENSUS -eq 1 ]; then
        log_substep "STEP 4: Creating consensus predictions"
        log_timestamp "START Step 4: Consensus predictions"
        local step4_start=$(date +%s)

        # Check if consensus files already exist
        consensus_test_file="${target_output_dir}/consensus/test_predictions.csv"
        consensus_global_file="${target_output_dir}/consensus/global_test_predictions.csv"

        if [ ! -f "$consensus_test_file" ] || [ ! -f "$consensus_global_file" ]; then
            echo "Generating consensus predictions from individual models..."
            echo "  This will create averaged predictions for all 2-model and 3-model combinations"

            python flaml-4-consensus.py "$target_output_dir"

            if [ $? -eq 0 ]; then
                local step4_end=$(date +%s)
                local step4_duration=$((step4_end - step4_start))
                log_timestamp "FINISH Step 4: Consensus predictions (duration: ${step4_duration}s / $((step4_duration/60))m)"
                echo "✓ Consensus predictions generated successfully"
                echo ""
                echo "Generated files:"
                echo "  - Consensus test predictions: $consensus_test_file"
                echo "  - Consensus global predictions: $consensus_global_file"
                echo "  - Consensus results: ${target_output_dir}/consensus/results.csv"
            else
                log_timestamp "FAILED Step 4: Consensus predictions"
                echo "✗ Consensus generation failed"
                return 1
            fi
        else
            echo "⊘ Consensus files already exist, skipping"
            check_file_exists "$consensus_test_file" "Consensus test predictions"
            check_file_exists "$consensus_global_file" "Consensus global predictions"
            log_timestamp "SKIP Step 4: Consensus predictions (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 4 (consensus predictions) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # STEP 5: Semi-supervised learning
    # ============================================================================
    if [ $RUN_SEMISUPERVISED -eq 1 ]; then
        log_substep "STEP 5: Semi-supervised learning with unlabeled data"
        log_timestamp "START Step 5: Semi-supervised learning"
        local step5_start=$(date +%s)

        # Check if semi-supervised results already exist
        semisup_results="${target_output_dir}/semisupervised/results.csv"

        if [ ! -f "$semisup_results" ]; then
            echo "Training models with semi-supervised learning..."
            echo "  Strategy: $semisup_strategy"
            echo "  Top-K models: $semisup_topk"
            echo "  Confidence threshold: $semisup_confidence"
            echo "  Agreement threshold: $semisup_agreement"

            # Check if we have the necessary files
            if [ ! -f "$best_configs_json" ]; then
                echo "✗ Error: Best configs file not found: $best_configs_json"
                echo "  Cannot run semi-supervised learning without FLAML configs"
                log_timestamp "FAILED Step 5: Semi-supervised learning (configs not found)"
                return 1
            fi

            # global_test_file is already defined at the top of dothings()
            if [ ! -f "$global_test_file" ]; then
                echo "✗ Error: Global test file (unlabeled data) not found: $global_test_file"
                echo "  Semi-supervised learning requires unlabeled data"
                log_timestamp "FAILED Step 5: Semi-supervised learning (unlabeled data not found)"
                return 1
            fi

            echo "  Unlabeled data: $global_test_file"
            echo ""

            # Run semi-supervised training with explicit file paths
            python flaml-4-train_semisupervised.py "$target_output_dir" \
                --strategy "$semisup_strategy" \
                --top-k "$semisup_topk" \
                --confidence "$semisup_confidence" \
                --agreement "$semisup_agreement" \
                --iterations "$semisup_iterations" \
                --samples-per-iter "$semisup_samples_per_iter" \
                --train-file "$train_file" \
                --test-file "$test_file" \
                --global-test-file "$global_test_file" \
                --random-state 42

            if [ $? -eq 0 ]; then
                local step5_end=$(date +%s)
                local step5_duration=$((step5_end - step5_start))
                log_timestamp "FINISH Step 5: Semi-supervised learning (duration: ${step5_duration}s / $((step5_duration/60))m)"
                echo "✓ Semi-supervised learning completed successfully"
                echo ""
                echo "Generated files:"
                echo "  - Semi-supervised results: $semisup_results"
                echo "  - Semi-supervised test predictions: ${target_output_dir}/semisupervised/test_predictions.csv"
                echo "  - Semi-supervised global predictions: ${target_output_dir}/semisupervised/global_test_predictions.csv"
            else
                log_timestamp "FAILED Step 5: Semi-supervised learning"
                echo "✗ Semi-supervised learning failed"
                return 1
            fi
        else
            echo "⊘ Semi-supervised results already exist, skipping"
            echo "  Results file: $semisup_results"
            log_timestamp "SKIP Step 5: Semi-supervised learning (already exists)"
        fi
        echo ""
    else
        echo "⊘ Skipping STEP 5 (semi-supervised learning) - disabled in configuration"
        echo ""
    fi

    # ============================================================================
    # Summary
    # ============================================================================
    local target_end_time=$(date +%s)
    local target_duration=$((target_end_time - target_start_time))

    log_step "COMPLETED: ${target_name}"
    log_timestamp "FINISH processing target: ${target_name} (total duration: ${target_duration}s / $((target_duration/60))m)"
    echo "All outputs are in: $target_output_dir"
    echo ""
    echo "Directory structure:"
    tree -L 2 "$target_output_dir" 2>/dev/null || ls -R "$target_output_dir"
    echo ""
    echo "To review performance:"
    echo "  cat ${target_output_dir}/results.csv"
    if [ $RUN_CONSENSUS -eq 1 ]; then
        echo "  cat ${target_output_dir}/consensus/results.csv"
    fi
    if [ $RUN_SEMISUPERVISED -eq 1 ]; then
        echo "  cat ${target_output_dir}/semisupervised/results.csv"
    fi
    echo ""
    echo "================================================================================"
    echo ""
}

# Export function and variables so they can be used by parallel
export -f dothings
export -f log_step
export -f log_substep
export -f log_timestamp
export -f check_file_exists

# Export configuration variables
export DESCRIPTOR_NAME
export DATA_BASE_DIR
export OUTPUT_BASE_DIR
export EXPERIMENT_SUFFIX
export GLOBAL_TEST_NAME
export budget
export n_best
export USE_WEIGHTS
export WEIGHT_STRATEGY
export output_base
export dir  # Legacy compatibility
export desc_prefix  # Legacy compatibility
export semisup_strategy
export semisup_topk
export semisup_confidence
export semisup_agreement
export semisup_iterations
export semisup_samples_per_iter

# Export control flags
export RUN_OPTIMIZATION
export RUN_EXTRACTION
export RUN_TRAINING
export RUN_TRAIN_OOF
export RUN_CONSENSUS
export RUN_SEMISUPERVISED
export RUN_SUMMARIZE

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log_step "FLAML COMPLETE WORKFLOW (CLEAN STRUCTURE WITH EXPLICIT PATHS)"
log_timestamp "START complete workflow"
workflow_start_time=$(date +%s)

echo "Configuration:"
echo "  Data base directory: $DATA_BASE_DIR"
echo "  Output base directory: $OUTPUT_BASE_DIR"
echo "  Descriptor set: $DESCRIPTOR_NAME"
echo "  Experiment suffix: ${EXPERIMENT_SUFFIX:-none}"
echo "  Global test dataset: $GLOBAL_TEST_NAME"
echo "  Output directory: $output_base"
echo "  FLAML time budget: ${budget} minutes"
echo "  Top N methods: ${n_best}"
echo "  Sample weighting: $([ $USE_WEIGHTS -eq 1 ] && echo "ENABLED (strategy: $WEIGHT_STRATEGY)" || echo "DISABLED")"
echo "  Parallel jobs: ${jobs}"
echo ""
echo "Targets to process:"
for target in "${TARGETS[@]}"; do
    echo "  - $target"
done
echo ""
echo "Workflow steps:"
echo "  1. FLAML optimization          [$([ $RUN_OPTIMIZATION -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  2. Extract best configs        [$([ $RUN_EXTRACTION -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  3. Train best models           [$([ $RUN_TRAINING -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  3A. OOF training predictions   [$([ $RUN_TRAIN_OOF -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  4. Consensus predictions       [$([ $RUN_CONSENSUS -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  5. Semi-supervised learning    [$([ $RUN_SEMISUPERVISED -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo "  6. Summarize & create final submission [$([ $RUN_SUMMARIZE -eq 1 ] && echo "ENABLED" || echo "DISABLED")]"
echo ""
echo "Semi-supervised configuration:"
echo "  Strategy: $semisup_strategy"
echo "  Top-K models: $semisup_topk"
echo "  Confidence threshold: $semisup_confidence"
echo "  Agreement threshold: $semisup_agreement"
echo "  Iterations: $semisup_iterations"
echo "  Samples per iteration: $semisup_samples_per_iter"
echo ""
log_step ""

# Process each target explicitly using the TARGETS array
echo "Processing ${#TARGETS[@]} target dataset(s)..."
echo ""

for target_name in "${TARGETS[@]}"; do
    dothings "$target_name"
done

# Alternative: Use parallel for production (uncomment to enable)
# Requires exporting TARGETS array
# printf "%s\n" "${TARGETS[@]}" | parallel --jobs "$jobs" --delay 60 --progress --eta dothings {}

echo ""
log_step "ALL DATASETS COMPLETED"

# ============================================================================
# STEP 6: Create final summary and best ensemble submission
# ============================================================================
if [ $RUN_SUMMARIZE -eq 1 ]; then
    log_step "STEP 6: Creating final summary and best ensemble submission"
    log_timestamp "START Step 6: Final summary"
    step6_start=$(date +%s)

    echo "Analyzing all results and selecting best methods for each target..."
    echo "  Output directory: $output_base"
    echo ""

    python flaml-9-summarize.py "$output_base"

    if [ $? -eq 0 ]; then
        step6_end=$(date +%s)
        step6_duration=$((step6_end - step6_start))
        log_timestamp "FINISH Step 6: Final summary (duration: ${step6_duration}s)"
        echo ""
        echo "✓ Summary and final submission created successfully"
        echo ""
        echo "Generated files:"
        echo "  - Complete summary: ${output_base}/final_summary.txt"
        echo "  - Best ensemble submission: ${output_base}/final_submission.csv"
        echo ""
        echo "The summary includes:"
        echo "  1. Best method for each of 4 targets (based on test AUROC)"
        echo "  2. Comparison of individual models, consensus, and semi-supervised"
        echo "  3. Final submission file ready for challenge upload"
    else
        log_timestamp "FAILED Step 6: Final summary"
        echo "✗ Summary generation failed"
    fi
    echo ""
else
    echo "⊘ Skipping STEP 6 (final summary) - disabled in configuration"
    echo ""
fi

workflow_end_time=$(date +%s)
workflow_duration=$((workflow_end_time - workflow_start_time))

echo ""
echo "================================================================================"
echo "FINAL SUMMARY"
echo "================================================================================"
log_timestamp "FINISH complete workflow (total duration: ${workflow_duration}s / $((workflow_duration/60))m / $((workflow_duration/3600))h)"
echo ""
echo "Results are in: $output_base"
echo ""
echo "Directory structure:"
echo ""
tree -L 2 "$output_base" 2>/dev/null || ls -R "$output_base"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review individual model results for each target:"
echo "     cat ${output_base}/{target}/results.csv"
echo ""
if [ $RUN_CONSENSUS -eq 1 ]; then
    echo "  2. Review consensus model results:"
    echo "     cat ${output_base}/{target}/consensus/results.csv"
    echo ""
fi
if [ $RUN_SEMISUPERVISED -eq 1 ]; then
    echo "  3. Review semi-supervised results:"
    echo "     cat ${output_base}/{target}/semisupervised/results.csv"
    echo ""
fi
if [ $RUN_SUMMARIZE -eq 1 ]; then
    echo "  4. AUTOMATED: Best ensemble submission created automatically"
    echo "     File: ${output_base}/final_submission.csv"
    echo "     Summary: ${output_base}/final_summary.txt"
    echo ""
    echo "  5. Submit to EUOS Challenge:"
    echo "     RECOMMENDED: ${output_base}/final_submission.csv"
else
    echo "  4. Create final ensemble (run step 6):"
    echo "     Set RUN_SUMMARIZE=1 and re-run this script"
fi
echo ""
log_step ""
