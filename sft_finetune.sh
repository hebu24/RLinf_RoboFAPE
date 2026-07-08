cd /opt/yingxi/RLinf_RoboFAPE && \
export PATH=/opt/kairan/envs/rlinf/bin:$PATH && \
CUDA_VISIBLE_DEVICES=2,5,6,7 bash examples/sft/run_vla_sft.sh peg_insertion_sft_openpi_pi05
