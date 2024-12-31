export HF_ENDPOINT=https://hf-mirror.com

python main.py \
    --model NCF-LLM \
    --batch-size 96 \
    --n-epochs 300 \
    --lr 2e-5 \
    --wd 0 \
    --optimizer adam \
    --device 1 \
    --eval-step 1 \
    --esp 7 \
    --exp-dir="./exp" \
    --exp-name="NCF4" \
    --emb-dim 1024 \
    --llm gpt2 \
    --loss bpr