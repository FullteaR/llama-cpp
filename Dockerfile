FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST "8.6"

RUN apt update && apt upgrade -y && apt -y autoremove
RUN apt install pciutils git build-essential cmake curl libcurl4-openssl-dev -y && apt -y autoremove
RUN git clone https://github.com/ggml-org/llama.cpp
RUN cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
RUN cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split llama-server
RUN cp llama.cpp/build/bin/llama-* llama.cpp

# CMD ./llama.cpp/llama-server -hf unsloth/gpt-oss-120b-GGUF:F16 --threads 28 --ctx-size 16384 --n-gpu-layers 999 -ot ".ffn_(up)_exps.=CPU" --temp 1.0 --min-p 0.0 --top-p 1.0 --top-k 0.0 --jinja --host 0.0.0.0 --port 18080 --alias gpt-oss-120b --chat-template-file /home/rikuta/chat_template.txt


CMD ./llama.cpp/llama-server -hf unsloth/gpt-oss-120b-GGUF:F16 --threads 28 --ctx-size 16384 --n-gpu-layers 999 -ot ".ffn_(up|down)_exps.=CPU" --temp 1.0 --min-p 0.0 --top-p 1.0 --top-k 0.0 --jinja --host 0.0.0.0 --port 18080 --alias gpt-oss-120b -np 4
# CMD ./llama.cpp/llama-server -hf unsloth/gpt-oss-20b-GGUF:F16 --threads 28 --ctx-size 32768 --n-gpu-layers 999 --temp 1.0 --min-p 0.0 --top-p 1.0 --top-k 0.0 --jinja --host 0.0.0.0 --port 18080 --alias gpt-oss-20b -np 4
