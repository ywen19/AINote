# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertModel, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import pickle
from REModel import REModel


MODEL_CLASSES = {
    'bert': BertModel,
    'roberta': RobertaModel,
}


IGNORE_INDEX = -100

        

class MyDataset():
    def __init__(self, prefix, data_path, h_t_limit):
        self.h_t_limit = h_t_limit

        self.data_path = data_path
        self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_train_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
        # question: data_train_bert_mask代表什么？
        self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_train_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


    def __getitem__(self, index):
        return self.train_file[index], self.data_train_bert_token[index],   \
                self.data_train_bert_mask[index], self.data_train_bert_starts_ends[index]
 
    def __len__(self):
        return self.data_train_bert_token.shape[0]

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class Config(object):
    def __init__(self, args):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.args = args

        self.max_seq_length = args.max_seq_length
        self.relation_num = 97

        self.max_epoch = args.num_train_epochs

        self.evaluate_during_training_epoch = args.evaluate_during_training_epoch

        self.log_period = args.logging_steps
        # question: 这里的neg_multiple代表什么？
        self.neg_multiple = 3
        self.warmup_ratio = 0.1

        self.data_path = args.prepro_data_dir
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.lr = args.learning_rate
        # question: 为什么要设置一个h_t_limit
        self.h_t_limit = 1800

        self.test_batch_size = self.batch_size * 2
        self.test_relation_limit = self.h_t_limit
        # question: 这里的self.dis2idx代表什么？
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'


        if not os.path.exists("log"):
            os.mkdir("log")

        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
            
        if not os.path.exists("fig_result"):
            os.mkdir("fig_result")

    def load_test_data(self):
        print("Reading testing data...")
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}

        prefix = self.test_prefix
        print (prefix)
        self.is_test = ('test' == prefix)
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_test_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_test_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


        self.test_len = self.data_test_bert_token.shape[0]
        assert(self.test_len==len(self.test_file))

        print("Finish reading")

        self.test_batches = self.data_test_bert_token.shape[0] // self.test_batch_size
        if self.data_test_bert_token.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_bert_token[x] > 0), reverse=True)

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()
        # question: h_mapping和t_mapping有什么区别？
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()

        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()

        context_masks = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()
   
        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()
            

            ht_pair_pos.zero_()

            # question: max_h_t_cnt有什么作用？
            max_h_t_cnt = 1

            labels = []

            L_vertex = []
            titles = []
            indexes = []

            evi_nums = []
            all_test_idxs = []

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_token[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]
                starts_pos = self.data_test_bert_starts_ends[index, :, 0]
                ends_pos = self.data_test_bert_starts_ends[index, :, 1]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])


                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                # question: 本代码是如何处理token数量大于512的文档的？
                test_idxs = []
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                            tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                            if len(hlist)==0 or len(tlist)==0:
                                continue
                            # question: 为什么要先除以len(hlist)， 再除以 (h[1] - h[0])？
                            for h in hlist:
                                h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                            for t in tlist:
                                t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                            # question: relation_mask的作用是什么？
                            relation_mask[i, j] = 1

                            delta_dis = hlist[0][0] - tlist[0][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                            test_idxs.append((h_idx, t_idx))
                            j += 1


                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}

                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in_annotated_train']

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)
                all_test_idxs.append(test_idxs)


            max_c_len = self.max_seq_length

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                #    'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'all_test_idxs': all_test_idxs,
                   }
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_train_batch(self, batch):
        batch_size = len(batch)
        max_length = self.max_seq_length
        h_t_limit = self.h_t_limit
        relation_num = self.relation_num
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        relation_multi_label = torch.Tensor(batch_size, h_t_limit, relation_num).zero_()
        relation_mask = torch.Tensor(batch_size, h_t_limit).zero_()

        context_masks = torch.LongTensor(batch_size, self.max_seq_length).zero_()
        ht_pair_pos = torch.LongTensor(batch_size, h_t_limit).zero_()

        relation_label = torch.LongTensor(batch_size, h_t_limit).fill_(IGNORE_INDEX)

        for i, item in enumerate(batch):
            max_h_t_cnt = 1

            context_idxs[i].copy_(torch.from_numpy(item[1]))
            context_masks[i].copy_(torch.from_numpy(item[2]))
            starts_pos = item[3][:, 0]
            ends_pos = item[3][:, 1]

            ins = item[0]
            labels = ins['labels']
            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])


            train_tripe = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_tripe:
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0][0] - tlist[0][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])


                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]

                j += 1


            lower_bound = min(len(ins['na_triple']), len(train_tripe) * self.neg_multiple)
            sel_idx = random.sample(list(range(len(ins['na_triple']))), lower_bound)
            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]

            for (h_idx, t_idx) in sel_ins:
                if j == h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])


                delta_dis = hlist[0][0] - tlist[0][0]

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_mask[i, j] = 1

                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                j += 1
            # question: max_h_t_cnt代表什么？
            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label[:, :max_h_t_cnt].contiguous(),
                'relation_multi_label': relation_multi_label[:, :max_h_t_cnt].contiguous(),
                'relation_mask': relation_mask[:, :max_h_t_cnt].contiguous(),
                'ht_pair_pos': ht_pair_pos[:, :max_h_t_cnt].contiguous(),
                'context_masks': context_masks,
                }

    def train(self, model_type, model_name_or_path, save_name):
        self.load_test_data()

        train_dataset = MyDataset(self.train_prefix, self.data_path, self.h_t_limit)
        #train_sampler = RandomSampler(train_dataset)
        #train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, collate_fn=self.get_train_batch, num_workers=2)

        # take subsets of data
        subset_size = int(len(train_dataset)*0.5)
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        subset_dataset = Subset(train_dataset, subset_indices)
        train_sampler = RandomSampler(subset_dataset)
        train_dataloader = DataLoader(subset_dataset, sampler=train_sampler, batch_size=self.batch_size,
                                      collate_fn=self.get_train_batch, num_workers=2)

        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)

        ori_model = REModel(config = self, bert_model=bert_model)
        ori_model.cuda()

        model = nn.DataParallel(ori_model)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.args.adam_epsilon)
        tot_step = int( (len(subset_dataset) // self.batch_size+1) / self.gradient_accumulation_steps * self.max_epoch)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.warmup_ratio*tot_step), num_training_steps=tot_step)

        save_step = int( (len(subset_dataset) // self.batch_size+1) / self.gradient_accumulation_steps  * self.evaluate_during_training_epoch)
        print("tot_step:", tot_step, "save_step:", save_step, self.lr)
        # question: 这里可以使用cross-entropy loss吗？
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)


        best_all_f1 = 0.0
        best_all_auc = 0
        best_all_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        step = 0

        # log metrics on train set for diagrams
        all_loss = []
        all_acc = []
        all_na_acc = []
        all_not_na_acc = []

        # log metrics on test set for diagrams as the training goes
        test_all_f1 = []
        test_all_ign_f1 = []
        test_all_auc = []

        for epoch in range(self.max_epoch):

            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for batch in train_dataloader:
                data = {k: v.cuda() for k,v in batch.items()}

                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']

                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']


                if torch.sum(relation_mask)==0:
                    print ('zero input')
                    continue
 
                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                pred_loss = BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)

                loss = torch.sum(pred_loss) /  (self.relation_num * torch.sum(relation_mask))
                if torch.isnan(loss):
                    pickle.dump(data, open("crash_data.pkl","wb"))
                    path = os.path.join(self.checkpoint_dir, model_name+"_crash")
                    torch.save(ori_model.state_dict(), path)


                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                loss.backward()
                all_loss.append(float(loss))

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label < 0:
                            break

                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                all_acc.append(self.acc_total.get())
                all_na_acc.append(self.acc_NA.get())
                all_not_na_acc.append(self.acc_not_NA.get())

                total_loss += loss.item()


                if (step + 1) % self.gradient_accumulation_steps == 0:            

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if global_step % self.log_period == 0 :
                        print("LOG Performance on Test Set...")
                        cur_loss = total_loss / self.log_period
                        elapsed = time.time() - start_time
                        # question: 这里的NA acc / not NA acc / tot acc分表代表什么？是如何计算的？
                        logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:.8f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                            epoch, global_step, elapsed * 1000 / self.log_period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                        total_loss = 0
                        start_time = time.time()



                    if global_step % save_step == 0:
                        logging('-' * 89)
                        eval_start_time = time.time()
                        model.eval()
                        all_f1, ign_f1, f1, auc, pr_x, pr_y = self.test(model, save_name)

                        # test
                        test_all_auc.append(auc)
                        test_all_f1.append(f1)
                        test_all_ign_f1.append(ign_f1)

                        model.train()
                        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                        logging('-' * 89)

                        if all_f1 > best_all_f1:
                            best_all_f1 = all_f1
                            best_all_epoch = epoch
                            best_all_auc = auc
                            path = os.path.join(self.checkpoint_dir, save_name)
                            torch.save(ori_model.state_dict(), path)
                            print("Storing result...")


                step += 1


        logging('-' * 89)
        eval_start_time = time.time()
        model.eval()
        all_f1, ign_f1, f1, auc, pr_x, pr_y = self.test(model, save_name)
        model.train()
        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
        logging('-' * 89)

        if all_f1 > best_all_f1:
            best_all_f1 = all_f1
            best_all_epoch = epoch
            path = os.path.join(self.checkpoint_dir, save_name)
            torch.save(ori_model.state_dict(), path)
            print("Storing result...")

        print("Finish training")
        print("Best epoch = %d | F1 = %f AUC = %f" % (best_all_epoch, best_all_f1, best_all_auc))

        # Train loss curve
        plt.figure()
        x = [i for i in range(1, len(all_loss)+1)]
        plt.xlabel('Itr')
        plt.ylim(0, 0.3)
        plt.plot(x, all_loss, linestyle='-', label='train')
        plt.legend()
        plt.title('Train Loss: Batch 8, Lr:4e-5')
        plt.savefig('./diagrams/subset5_train_loss_batch8_lr4e-5.png')

        # train acc
        plt.figure()
        plt.xlabel('Itr')
        plt.plot(x, all_acc, linestyle='-', label='All Acc')
        plt.plot(x, all_na_acc, linestyle='-', label='Acc for NA Relation')
        plt.plot(x, all_not_na_acc, linestyle='-', label='Acc for Not-NA Relation')
        plt.legend()
        plt.title('Train Acc: Batch 8, Lr:4e-5')
        plt.savefig('./diagrams/subset5_train_acc_batch8_lr4e-5.png')

        # f1 and auc on test set
        plt.figure()
        x = [i*20 for i in range(1, len(test_all_auc) + 1)]
        plt.xlabel('Itr')
        plt.plot(x, test_all_f1, linestyle='-', label='F1')
        plt.plot(x, test_all_ign_f1, linestyle='-', label='F1(Data unseen in train)')
        plt.legend()
        plt.title('F1 on test set: Batch 8, Lr:4e-5')
        plt.savefig('./diagrams/subset5_f1_batch8_lr4e-5.png')

        # f1 and auc on test set
        plt.figure()
        plt.xlabel('Epoch')
        plt.plot(x, test_all_auc, linestyle='-', label='AUC')
        plt.legend()
        plt.title('AUC on test set: Batch 8, Lr:4e-5')
        plt.savefig('./diagrams/subset5_auc_batch8_lr4e-5.png')


    def test(self, model, save_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        predicted_as_zero = 0
        total_ins_num = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                L_vertex = data['L_vertex']
                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']
                all_test_idxs = data['all_test_idxs']


                titles = data['titles']
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)
                # question: 这里是否可以改成softmax函数？
                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()
            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]


                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                test_idxs = all_test_idxs[i]
                j = 0

                for (h_idx, t_idx) in test_idxs:
                    # get the most likely relationship from prediction
                    r = np.argmax(predict_re[i, j])
                    predicted_as_zero += (r==0)
                    total_ins_num += 1
                    # if tuple(h,t,predict_r) in data
                    if (h_idx, t_idx, r) in label:
                        top1_acc += 1

                    flag = False
                    # relation_num: in total 97 relations are given for classification
                    for r in range(1, self.relation_num):
                        intrain = False

                        if (h_idx, t_idx, r) in label:
                            flag = True
                            if label[(h_idx, t_idx, r)]==True:
                                intrain = True
                        test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                    if flag:
                        have_label += 1

                    j += 1

            data_idx += 1

            if data_idx % self.log_period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.log_period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        test_result.sort(key = lambda x: x[1], reverse=True)

        print ('total_recall', total_recall)
        print('predicted as zero', predicted_as_zero)
        print('total ins num', total_ins_num)
        print('top1_acc', top1_acc)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            # correct: count of the predicted tuples that actually occur in the test set
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i


        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max() # max f1 score
        f1_pos = f1_arr.argmax()
        all_f1 = f1
        theta = test_result[f1_pos][1] #probability when the f1 score reaches its highest

        if input_theta==-1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if not self.is_test:
            # question: 这里的Theta / F1 / AUC分别代表什么？是如何计算的？
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            # question: 这里的input_theta / test_result F1 / AUC分别代表什么？是如何计算的？
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        if output:
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(save_name + "_" + self.test_prefix + "_index.json", "w"))
            print ('finish output')

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        # question: 这里的input_theta / test_result F1 / AUC分别代表什么？是如何计算的？
        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(ign_f1, input_theta, f1_arr[w], auc))
        
        return all_f1, ign_f1, f1_arr[w], auc, pr_x, pr_y


    def testall(self, model_type, model_name_or_path, save_name, input_theta): 
        self.load_test_data()
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)
        model = REModel(config = self, bert_model=bert_model)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, save_name)))
        model.cuda()
        model = nn.DataParallel(model)
        model.eval()

        self.test(model, save_name, True, input_theta)
