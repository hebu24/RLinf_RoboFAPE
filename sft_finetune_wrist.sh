cd /opt/yingxi/RLinf_RoboFAPE && \
export PATH=/opt/kairan/envs/rlinf/bin:$PATH && \
export RAY_TMPDIR=/opt/yingxi/RLinf_RoboFAPE/ray_tmp && \
export CUDA_LAUNCH_BLOCKING=1 && \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/sft/run_vla_sft.sh \
  peg_insertion_sft_openpi_pi05_wrist \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}"
  