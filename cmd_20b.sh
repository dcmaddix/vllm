vllm serve openai/gpt-oss-20b --trust-remote-code --enable-lora --max-loras 1\
 --lora-modules \
 lora1=/home/ubuntu/model/gpt-oss-20b-loraadapter/lora_adapter \
 --max-lora-rank 16


# vllm serve /home/ubuntu/models/gpt-oss-20b/base_model --trust-remote-code \
#  --enable-lora --max-loras 1\
#  --lora-modules \
#  lora1=/home/ubuntu/models/gpt-oss-20b/moe-all-eager/lora_adapter \
#  --max-lora-rank 16
