import argparse
from os.path import join
from tqdm import tqdm
from transformers import BertModel
from transformers import BertConfig as BC

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper
from data_process import InputFeatures, Example
from evaluate import eval_performance
from torch.cuda.amp import GradScaler
# apex conflicts with virtual env, use torch amp instead, and is more flexible
"""try:
    from apex import amp
except Exception:
    print('Apex not imoport!')"""


import torch
from torch import nn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch

def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    # for joint task training
    # loss = w1*loss1 + w2*loss2 + w3*loss3 + ...
    # loss1: for answer span
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    # loss2: for answer type, type_lambda is a weight
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])
    # loss3: for support facts
    sent_num_in_batch = batch["start_mapping"].sum()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3



import json

@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()

    for batch in tqdm(dataloader):
        total_test_loss = [0] * 5

        batch['context_mask'] = batch['context_mask'].float()
        # start_logits: (batch_size, sequence_length) -> prediction logits for start of answer span
        # end_logits: (batch_size, sequence_length) -> prediction logits for end of answer span
        # type_logits: (batch_size, answer_types=4) -> prediction logits for answer type
        # sp_logits: (batch_size, sent count) -> probability if the current sentence is support fact
        # start_position: (batch_size) ->  predicted start indices of the answer span
        # end_position: (batch_size) -> predicted end indices of the answer span
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)

        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()
        for i in range(len(total_test_loss)):
            total_test_loss[i] /= len(batch)
        update_losses_to_record_dict(dev_loss_record, total_test_loss)

        # get the predicted answers {"doc_id": answer}
        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()  # sigmoid for binary classification
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):  # sent_names: (title, local_sent_id)
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    # threshold to eliminate support facts that have too low probability (noise reduction)
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict = {}
    for key, value in answer_dict.items():
        new_answer_dict[key] = value.replace(" ", "")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)

    for i in range(len(total_test_loss)):
        print("Test Loss{}: {}".format(i, total_test_loss[i]))
    return total_test_loss


def train_epoch(data_loader, model, predict_during_train=False):
    model.train()
    pbar = tqdm(total=len(data_loader))
    epoch_len = len(data_loader)
    step_count = 0
    predict_step = epoch_len // 5

    while not data_loader.empty():
        step_count += 1
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float()
        batch_train_loss = train_batch(model, batch)
        del batch

        # check model performance on dev set every 5(default) steps, save model and prediction
        if predict_during_train and (step_count % predict_step == 0):
            dev_pred_file_path = join(
                args.prediction_path, 'pred_dev_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)
            )
            predict(model, eval_dataset, dev_example_dict, dev_feature_dict, dev_pred_file_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pth".format(args.seed, epc, step_count)))
            model.train()
        pbar.update(1)

    # check model performance on the train set after an epoch is done
    train_pred_file_path = join(
        args.prediction_path, 'pred_train_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)
    )
    total_train_loss = predict(model, data_loader, train_example_dict, train_feature_dict, train_pred_file_path)
    train_metrics = eval_performance(train_pred_file_path, "../data/final_train.json")

    # save model and prediction result on dev set after the current epoch is done
    dev_pred_file_path = join(
        args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)
    )
    epoch_test_losses = predict(model, eval_dataset, dev_example_dict, dev_feature_dict, dev_pred_file_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pth".format(args.seed, epc)))
    dev_metrics = eval_performance(dev_pred_file_path, "../data/final_dev.json")

    return train_metrics, dev_metrics


def train_batch(model, batch):
    global global_step

    batch_train_loss = [0]*5

    start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
    loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
    loss_list = list(loss_list)
    if args.gradient_accumulation_steps > 1:
        loss_list[0] = loss_list[0] / args.gradient_accumulation_steps
    
    if args.fp16:
        with torch.autocast(device_type="cuda"):
            scaler.scale(loss_list[0]).backward()
        """with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
            scaled_loss.backward()"""
    else:
        loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            batch_train_loss[i] += l.item()

    # average the loss
    for i in range(len(batch_train_loss)):
        batch_train_loss[i] /= VERBOSE_STEP
    # update the loss record
    update_losses_to_record_dict(train_loss_record, batch_train_loss)

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(batch_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l))
    return batch_train_loss


def update_losses_to_record_dict(record_dict, losses):
    record_dict["loss"].append(losses[0])
    record_dict["span_loss"].append(losses[1])
    record_dict["type_loss"].append(losses[2])
    record_dict["sp_loss"].append(losses[3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    args.n_gpu = torch.cuda.device_count()

    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    train_example_dict = helper.train_example_dict
    train_feature_dict = helper.train_feature_dict
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader

    roberta_config = BC.from_pretrained(args.bert_model)  # {id: example}
    encoder = BertModel.from_pretrained(args.bert_model)  # {id: InputFeature}
    args.input_dim = roberta_config.hidden_size
    model = BertSupportNet(config=args, encoder=encoder)
    if args.trained_weight is not None:
        model.load_state_dict(torch.load(args.trained_weight))
    model.to('cuda')

    # Initialize optimizer and criterion
    lr = args.lr
    # calculate the warmup steps and training steps for scheduled update
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = 0.1 * t_total
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  

    """if args.fp16:
        import apex
        apex.amp.register_half_function(torch, "einsum")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')"""

    global scaler
    scaler = GradScaler()

    model = torch.nn.DataParallel(model)
    model.train()

    # Training

    # metrics to record model performances
    performance_on_train = {}
    performance_on_dev = {}

    global_step = epc = 0
    # loss record on per epoch
    train_loss_record = {"loss": [], "span_loss": [], "type_loss": [], "sp_loss": []}
    dev_loss_record = {"loss": [], "span_loss": [], "type_loss": [], "sp_loss": []}
    VERBOSE_STEP = args.verbose_step

    for epc in range(1, args.epochs+1):
        Loader = Full_Loader
        Loader.refresh()

        train_metrics, dev_metrics = train_epoch(Loader, model, predict_during_train=True)
        performance_on_train[epc] = train_metrics
        performance_on_dev[epc] = dev_metrics
        print(performance_on_train, performance_on_dev)

    # save the record
    with open(join(args.prediction_path, "train_metrics.json"), "w") as f:
        json.dump(performance_on_train, f, indent=4)
    with open(join(args.prediction_path, "dev_metrics.json"), "w") as f:
        json.dump(performance_on_dev, f, indent=4)
    with open(join(args.prediction_path, "train_epoch_losses.json"), "w") as f:
        json.dump(train_loss_record, f, indent=4)
    with open(join(args.prediction_path, "dev_epoch_losses.json"), "w") as f:
        json.dump(dev_loss_record, f, indent=4)
