cd /opt/yingxi/RLinf_RoboFAPE && \
export PATH=/opt/kairan/envs/rlinf/bin:$PATH && \
export RAY_TMPDIR=/opt/yingxi/RLinf_RoboFAPE/ray_tmp && \
export CUDA_LAUNCH_BLOCKING=1 && \
DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200}" && \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/sft/run_vla_sft.sh \
  peg_insertion_sft_openpi_pi05 \
  data.train_data_paths="${DATA_DIR}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}"
