import os
import sys
import torch
import logging
import argparse
from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from trainer import Trainer
from recommender import NCFRecommender, LLMBasedNCFRecommender
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="RecLLM training.")
    
    parser.add_argument('--model', type=str, default='NCF', help="Model name.")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size.")
    parser.add_argument('--n-epochs', type=int, default=300, help="Epoch number.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--wd', type=float, default=.0, help="Weight decay.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer.')
    parser.add_argument('--device', type=int, default=0, help="GPU id, -1 for CPU.")
    parser.add_argument('--eval-step', type=int, default=1, help="Evaluation interval.")
    parser.add_argument('--esp', type=int, default=10, help="Early stop patience.")
    parser.add_argument('--exp-dir', type=str, default='./exp', help="Experiment directory.")
    parser.add_argument('--exp-name', type=str, default='NCF_1', help="Experiment directory.")
    parser.add_argument('--loss', type=str, default='pairwise', help="Name of loss function. Please refer to the name in loss.py.")
    
    parser.add_argument('--emb-dim', type=int, default=1024, help="Embedding dim.")
    parser.add_argument('--llm', type=str, default='gpt2', help="LLM model name.")

    args = parser.parse_args()
    
    args.device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.exp_path = f"{args.exp_dir}/{args.exp_name}/{current_time}" + \
        f"_lr{f'{args.lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')}"
    os.makedirs(args.exp_path, exist_ok=True)
    args.model_path = f"{args.exp_path}/ckpt.pth"
    
    with open(f"{args.exp_path}/args", mode='wt') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(
            torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> Args:\n')
        for k, v in sorted(vars(args).items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
    
    return args

def setup_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{args.exp_path}/log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
def present_result(name, result):
    print_str = f"{name}:\n"
    for k,v in result.items():
        print_str += f"    {k:<7}: {v:.4f}\n"
    logging.info(print_str)
    print(print_str, end='')
    
def create_model(args, dataset):
    if args.model in ['NCF']:
        module = NCFRecommender
        model_kwargs = {
            'embed_dim': args.emb_dim,
            'llm': 'none',
            'loss': args.loss
        }
    elif args.model in ['NCF-LLM']:
        module = LLMBasedNCFRecommender
        model_kwargs = {
            'embed_dim': args.emb_dim,
            'llm': args.llm,
            'loss': args.loss
        }
    else:
        raise NotImplementedError
    
    return module(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        user_meta_fn=dataset.get_user_meta,
        item_meta_fn=dataset.get_item_meta,
        **model_kwargs)

def train():
    args = parse_args()
    setup_logger(args)
    
    # Load dataset
    dataset = ML1MDataset('ml-1m')

    # Create model
    model = create_model(args, dataset)

    # Get split datasets
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')

    # Create dataloaders
    train_loader = TrainDataLoader(train_data, 
        batch_size=args.batch_size, shuffle=True, device=args.device)
    valid_loader = EvalDataLoader(valid_data, train_data, 
        batch_size=args.batch_size, device=args.device)
    test_loader = EvalDataLoader(test_data, train_data, 
        batch_size=args.batch_size, device=args.device)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        device=args.device,
        epochs=args.n_epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.wd,
        eval_step=args.eval_step,
        early_stop_patience=args.esp
    )

    valid_result, test_result = trainer.fit(save_model=True, model_path=args.model_path)
    present_result("Best Validation Result", valid_result)
    present_result("Test Result", test_result)
    
if __name__ == '__main__':
    train()