import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import Vocab, DataLoader, STR, END, CLS, SEL, TL, rCLS
from generator import Generator
from extract import LexicalMap
from adam import AdamWeightDecayOptimizer
from utils import move_to_cuda
from work import validate
import argparse, os
import datetime
import random
import re

def parse_config():
    parser = argparse.ArgumentParser()
    # vocabs
    parser.add_argument('--token_vocab', type=str)
    parser.add_argument('--concept_vocab', type=str)
    parser.add_argument('--predictable_token_vocab', type=str)
    parser.add_argument('--token_char_vocab', type=str)
    parser.add_argument('--concept_char_vocab', type=str)
    parser.add_argument('--relation_vocab', type=str)
    parser.add_argument('--pretrained_file', type=str, default=None)
    parser.add_argument('--log_file', type=str, default='train.log')

    # concept/token encoders
    parser.add_argument('--token_char_dim', type=int)
    parser.add_argument('--token_dim', type=int)
    parser.add_argument('--concept_char_dim', type=int)
    parser.add_argument('--concept_dim', type=int)

    # char-cnn
    parser.add_argument('--cnn_filters', type=int, nargs = '+')
    parser.add_argument('--char2word_dim', type=int)
    parser.add_argument('--char2concept_dim', type=int)

    # relation encoder
    parser.add_argument('--rel_dim', type=int)
    parser.add_argument('--rnn_hidden_size', type=int)

    # core architecture
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--snt_layers', type=int)
    parser.add_argument('--graph_layers', type=int)
    parser.add_argument('--inference_layers', type=int)

    # dropout/unk
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--unk_rate', type=float)

    # IO
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--dev_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)

    # distributed training
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)

    return parser.parse_args()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def update_lr(optimizer, embed_size, steps, warmup_steps):
    for param_group in optimizer.param_groups:
        param_group['lr'] = embed_size**-0.5 * min(steps**-0.5, steps*(warmup_steps**-1.5))

def main(args, local_rank):
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 5, [CLS])
    vocabs['token'] = Vocab(args.token_vocab, 5, [STR, END])
    vocabs['predictable_token'] = Vocab(args.predictable_token_vocab, 5, [END])
    vocabs['token_char'] = Vocab(args.token_char_vocab, 100, [STR, END])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [STR, END])
    vocabs['relation'] = Vocab(args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
    lexical_mapping = LexicalMap()

    for name in vocabs:
        print ((name, vocabs[name].size, vocabs[name].coverage))

    torch.manual_seed(19940117)
    torch.cuda.manual_seed_all(19940117)
    random.seed(19940117)

    device = torch.device('cuda', local_rank)
    model = Generator(vocabs,
            args.token_char_dim, args.token_dim,
            args.concept_char_dim, args.concept_dim,
            args.cnn_filters, args.char2word_dim, args.char2concept_dim,
            args.rel_dim, args.rnn_hidden_size,
            args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
            args.snt_layers, args.graph_layers, args.inference_layers,
            args.pretrained_file,
            device).cuda(device)

    if args.world_size > 1:
        torch.manual_seed(19940117 + dist.get_rank())
        torch.cuda.manual_seed_all(19940117 + dist.get_rank())
        random.seed(19940117+dist.get_rank())

    train_data = DataLoader(vocabs, lexical_mapping, args.train_data, args.train_batch_size, for_train=True)
    dev_data = DataLoader(vocabs, lexical_mapping, args.dev_data, args.dev_batch_size, for_train=False)
    train_data.set_unk_rate(args.unk_rate)

    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay':1e-4},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    optimizer = AdamWeightDecayOptimizer(grouped_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-6)

    epoch = 0
    batches_acm, loss_acm = 0, 0
    discarded_batches_acm = 0
    best_bleu = 0
    if hasattr(args, 'last_ckpt'):
        ckpt = torch.load(args.last_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        batches_acm = ckpt['batches_acm']
        loss_acm = ckpt['loss_acm']
        best_bleu = ckpt['bleu']
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        del ckpt

    with open(os.path.join(args.ckpt, args.log_file), 'a') as log_f:
        log_f.write('Start time: {}\n'.format(datetime.datetime.now()))
        log_f.flush()
        for epoch in range(epoch, args.epochs):
            model.train()
            for batch in train_data:
                batch = move_to_cuda(batch, device)
                loss = model(batch)
                loss_value = loss.item()
                if batches_acm > args.warmup_steps and loss_value > 5.*\
                        (loss_acm / batches_acm):
                    discarded_batches_acm += 1
                    print ('abnormal', loss_value)
                    continue
                loss_acm += loss_value
                batches_acm += 1
                loss.backward()
                if args.world_size > 1:
                    average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                update_lr(optimizer, args.embed_dim, batches_acm, args.warmup_steps)

                optimizer.step()
                optimizer.zero_grad()
                if args.world_size == 1 or (dist.get_rank() == 0):
                    if batches_acm % args.print_every == -1 % args.print_every:
                        print ('epoch %d, batch %d, discarded %d,\
                                loss %.3f' % (epoch, batches_acm,
                                    discarded_batches_acm, loss_acm/batches_acm))
                        model.train()
                    if batches_acm > args.warmup_steps and batches_acm %\
                            args.eval_every == -1 % args.eval_every:
                        model.eval()
                        bleu, chrf = validate(model, dev_data,
                                args.dev_data[:-len('.preproc.json')])
                        log_f.write('{:3} {:.4f} {}\n'.format(epoch, bleu, chrf))
                        log_f.flush()
                        if bleu > best_bleu:
                            best_bleu = bleu
                            ckpt_files = get_ckpt_files(args.ckpt)
                            if len(ckpt_files) > 2:
                                os.remove(min(ckpt_files, key=os.path.getctime))
                            torch.save({
                                'args': args,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch,
                                'batches_acm': batches_acm,
                                'loss_acm': loss_acm,
                                'bleu': bleu},
                                os.path.join(args.ckpt ,'batch{:06d}_epoch{:03d}'\
                                        .format(batches_acm, epoch)))
                        model.train()
        log_f.write('Finish time: {}\n'.format(datetime.datetime.now()))

def init_processes(args, local_rank, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank,
            world_size=args.world_size)
    main(args, local_rank)

def get_ckpt_files(ckpt_path):
    return [os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path) if\
            re.search(r'\d+$', x)]

if __name__ == "__main__":
    args = parse_config()
    if os.path.exists(args.ckpt):
        files = get_ckpt_files(args.ckpt)
        if files:
            args.last_ckpt = max(files, key=os.path.getctime)
    else:
        os.mkdir(args.ckpt)
    assert len(args.cnn_filters)%2 == 0
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))

    if args.world_size == 1:
        main(args, 0)
        exit(0)
    args.train_batch_size = args.train_batch_size / args.world_size
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    never_stop = False
    while never_stop:
        never_stop = True
