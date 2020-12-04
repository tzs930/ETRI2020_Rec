import os
import time
import argparse
import tensorflow as tf
from samplerAttr import WarpSampler
from modelAttr import Model
from tqdm import tqdm
from utilAttr import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', required=False)
parser.add_argument('--train_dir', default='default', required=False)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args3.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log3.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
valid_sampler = WarpSampler(user_valid, usernum, itemnum, batch_size=usernum, maxlen=args.maxlen, n_workers=1)
test_sampler = WarpSampler(user_test, usernum, itemnum, batch_size=usernum, maxlen=args.maxlen, n_workers=1)

model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):
        losses = []
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b', desc="Epoch %d"%epoch):
            u, seq, pos, neg, seq_attr, pos_attr, neg_attr= sampler.next_batch()
            
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.seq_attr: seq_attr, model.pos_attr: pos_attr, model.neg_attr: neg_attr,
                                     model.is_training: True})
            losses.append(loss)

        if epoch % 1 == 0:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',

            vu, vseq, vpos, vneg, vseq_attr, vpos_attr, vnegattr = valid_sampler.next_batch()
            tu, tseq, tpos, tneg, tseq_attr, tpos_attr, tnegattr = test_sampler.next_batch()
            
            _, vloss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: vu, model.input_seq: vseq, model.pos: vpos, model.neg: vneg,
                                     model.seq_attr: vseq_attr, model.pos_attr: vpos_attr, model.neg_attr: vnegattr,
                                     model.is_training: False})
            
            _, tloss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: tu, model.input_seq: tseq, model.pos: tpos, model.neg: tneg,
                                     model.seq_attr: tseq_attr, model.pos_attr: tpos_attr, model.neg_attr: tnegattr,
                                     model.is_training: False})

            train_loss = np.mean(losses)

            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)

            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, Loss: %.4f), test (NDCG@10: %.4f, HR@10: %.4f,  Loss: %.4f)' % \
                (epoch, T, t_valid[0], t_valid[1], vloss, t_test[0], t_test[1], tloss)

            f.write(str(t_valid) + ' ' + str(t_test)  + ' ' + str(train_loss) + ' ' + str(vloss) + ' '  +  str(tloss) + '\n')
            f.flush()
            t0 = time.time()
            
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
