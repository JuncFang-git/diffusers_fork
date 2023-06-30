###
 # @Author: Juncfang
 # @Date: 2023-02-24 10:30:06
 # @LastEditTime: 2023-03-02 19:41:58
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/script/wui2hf.sh
 #  
### 

export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export CKPT_PATH="/home/juncfang/code/diffusers_fork/personal_workspace/base_model/awportraitv1.1.ckpt"
export HF_DIR="/home/juncfang/code/diffusers_fork/personal_workspace/base_model/awportraitv1.1"
if [[ ! -d $HF_DIR ]]; then
    mkdir -p $HF_DIR
fi

python $PROJECT_DIR/scripts/convert_original_stable_diffusion_to_diffusers.py \
--checkpoint_path "$CKPT_PATH" \
--dump_path "$HF_DIR" \
# --to_safetensors \
# --from_safetensors \
# --original_config_file ./v1-inference.yaml \