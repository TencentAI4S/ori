import os, sys, glob
import argparse
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_path', type=str, default='', help='path to the diffusion model')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')

    parser.add_argument('--bsz', type=int, default=60, help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'test_glp1rgcgr'], help='dataset split used to decode')

    parser.add_argument('--top_p', type=int, default=1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    output_lst = []

    if args.split == 'test':
        out_dir = f'generation_outputs_{args.top_p}'
    elif args.split == 'test_glp1rgcgr':
        out_dir = f'generation_outputs_glp1rgcgr_{args.top_p}'

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    COMMAND = f'python -m torch.distributed.launch --nproc_per_node=1 --master_port={12233 + int(args.seed)} --use_env sample_seq2seq.py ' \
    f'--model_path {args.model_path} --step {args.step} ' \
    f'--batch_size {args.bsz} --seed2 {args.seed} --split {args.split} ' \
    f'--out_dir {out_dir} --top_p {args.top_p} '
    print(COMMAND)
            
    os.system(COMMAND)
    
    print('#'*30, 'decoding finished...')