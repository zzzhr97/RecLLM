export HF_ENDPOINT=https://hf-mirror.com

python main.py \
    --model NCF-LLM \
    --batch-size 512 \
    --n-epochs 300 \
    --lr 1e-4 \
    --wd 0 \
    --optimizer adam \
    --device 2 \
    --eval-step 1 \
    --esp 15 \
    --exp-dir="./exp" \
    --exp-name="NCF3" \
    --emb-dim 1024 \
    --llm gpt2