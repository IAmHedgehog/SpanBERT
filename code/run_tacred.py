import argparse
import logging
import os
import random
import time
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam


CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.org_id = guid.split('_')[0]
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DataProcessor(object):
    """Processor for the TACRED data set."""

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        return data

    def get_train_examples(self, train_file):
        """See base class."""
        return self._create_examples(self._read_json(train_file), "train")

    def get_dev_examples(self, dev_file):
        """See base class."""
        return self._create_examples(self._read_json(dev_file), "dev")

    def get_test_examples(self, test_file):
        """See base class."""
        raw_data = self._read_json(test_file)
        return self._create_examples(raw_data, "test"), np.array(raw_data)

    def get_labels(self, train_file, negative_label="no_relation"):
        """See base class."""
        dataset = self._read_json(train_file)
        count = Counter()
        for example in dataset:
            count[example['relation']] += 1
        logger.info("%d labels" % len(count))
        # Make sure the negative label is alwyas 0
        labels = [negative_label]
        for label, count in count.most_common():
            logger.info("%s: %.2f%%" % (label, count * 100.0 / len(dataset)))
            if label not in labels:
                labels.append(label)
        return labels

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = [convert_token(token) for token in example['token']]
            assert example['subj_start'] >= 0 and example['subj_start'] <= example['subj_end'] \
                and example['subj_end'] < len(sentence)
            assert example['obj_start'] >= 0 and example['obj_start'] <= example['obj_end'] \
                and example['obj_end'] < len(sentence)
            examples.append(InputExample(guid=example['id'],
                            sentence=sentence,
                            span1=(example['subj_start'], example['subj_end']),
                            span2=(example['obj_start'], example['obj_end']),
                            ner1=example['subj_type'],
                            ner2=example['obj_type'],
                            label=example['relation']))
        return examples


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, mode='text'):
    """Loads a data file into a list of `InputBatch`s."""
    special_tokens = {
        'SUBJ=ORGANIZATION': '[unused1]',
        'SUBJ=PERSON': '[unused2]',
        'OBJ=PERSON': '[unused3]',
        'OBJ=ORGANIZATION': '[unused4]',
        'OBJ=DATE': '[unused5]',
        'OBJ=NUMBER': '[unused6]',
        'OBJ=TITLE': '[unused7]',
        'OBJ=COUNTRY': '[unused8]',
        'OBJ=LOCATION': '[unused9]',
        'OBJ=CITY': '[unused10]',
        'OBJ=MISC': '[unused11]',
        'OBJ=STATE_OR_PROVINCE': '[unused12]',
        'OBJ=DURATION': '[unused13]',
        'OBJ=NATIONALITY': '[unused14]',
        'OBJ=CAUSE_OF_DEATH': '[unused15]',
        'OBJ=CRIMINAL_CHARGE': '[unused16]',
        'OBJ=RELIGION': '[unused17]',
        'OBJ=URL': '[unused18]',
        'OBJ=IDEOLOGY': '[unused19]'
    }
    kg = {}
    object_offset = 3

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    unique_eids = sorted(set([example.org_id for example in examples]))
    num_tokens = 0
    num_fit_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
        OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)
        subject_id, object_id = tokenizer.convert_tokens_to_ids([SUBJECT_NER, OBJECT_NER])
        relation_id = label2id[example.label]
        e1rel = (subject_id, relation_id)
        if e1rel not in kg:
            kg[e1rel] = set()
        # Subtract offset so that the JRRELP loss labels are indexed correctly
        kg[e1rel].add(object_id - object_offset)

        if mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_START)
                if i == example.span2[0]:
                    tokens.append(OBJECT_START)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                if i == example.span1[1]:
                    tokens.append(SUBJECT_END)
                if i == example.span2[1]:
                    tokens.append(OBJECT_END)
            if mode == "text_ner":
                tokens = tokens + [SEP, SUBJECT_NER, SEP, OBJECT_NER, SEP]
            else:
                tokens.append(SEP)
        else:
            subj_tokens = []
            obj_tokens = []
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_NER)
                if i == example.span2[0]:
                    tokens.append(OBJECT_NER)
                if (i >= example.span1[0]) and (i <= example.span1[1]):
                    for sub_token in tokenizer.tokenize(token):
                        subj_tokens.append(sub_token)
                elif (i >= example.span2[0]) and (i <= example.span2[1]):
                    for sub_token in tokenizer.tokenize(token):
                        obj_tokens.append(sub_token)
                else:
                    for sub_token in tokenizer.tokenize(token):
                        tokens.append(sub_token)
            if mode == "ner_text":
                tokens.append(SEP)
                for sub_token in subj_tokens:
                    tokens.append(sub_token)
                tokens.append(SEP)
                for sub_token in obj_tokens:
                    tokens.append(sub_token)
            tokens.append(SEP)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example.label]
        eid = unique_eids.index(example.org_id)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              subject_id=subject_id,
                              eid=eid))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels, eids):
    n_gold = n_pred = n_correct = 0
    aggreated = defaultdict(list)
    for pred, label, eid in zip(preds, labels, eids):
        aggreated[eid].append((pred, label))

    for eid, values in aggreated.items():
        cur_preds = [value[0] for value in values]
        cur_labels = [value[1] for value in values]
        if len(set(cur_labels)) != 1:
            print('=======>', eid, cur_labels)
        assert len(set(cur_labels)) == 1
        label = cur_labels[0]
        pred = max(set(cur_preds), key=cur_preds.count)

        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def compute_structure_parts(data):
    argdists = []
    sentlens = []
    for instance in data:
        ss, se = instance['subj_start'], instance['subj_end']
        os, oe = instance['obj_start'], instance['obj_end']
        sentlens.append(len(instance['token']))
        if ss > oe:
            argdist = ss - oe
        else:
            argdist = os - se
        argdists.append(argdist)
    return {'argdists': argdists, 'sentlens': sentlens}


def compute_structure_errors(parts, preds, gold_labels):
    structure_errors = {'argdist=1': [], 'argdist>10': [], 'sentlen>30': []}
    argdists = parts['argdists']
    sentlens = parts['sentlens']
    for i in range(len(argdists)):
        argdist = argdists[i]
        sentlen = sentlens[i]
        pred = preds[i]
        gold = gold_labels[i]
        is_correct = pred == gold

        if argdist <= 1:
            structure_errors['argdist=1'].append(is_correct)
        if argdist > 10:
            structure_errors['argdist>10'].append(is_correct)
        if sentlen > 30:
            structure_errors['sentlen>30'].append(is_correct)
    print('Structure Errors:')
    for structure_name, error_list in structure_errors.items():
        accuracy = round(np.mean(error_list) * 100., 4)
        print('{} | Accuracy: {} | Correct: {} | Wrong: {} | Total: {} '.format(
            structure_name, accuracy, sum(error_list), len(error_list) - sum(error_list), len(error_list)
        ))
    return structure_errors


def evaluate(model, device, eval_dataloader, eval_label_ids, eval_eids, num_labels, id2label, verbose=True, raw_data=None):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, eids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        # eids = eids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds[0], axis=1).reshape(-1)
    result = compute_f1(preds, eval_label_ids.numpy(), eval_eids.numpy())
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    processor = DataProcessor()
    label_list = processor.get_labels(args.train_file, args.negative_label)
    label2id = {label: i for i, label in enumerate(label_list)}
    print(label2id)

    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    special_tokens = {}
    if args.do_eval and not args.eval_test:
        eval_examples = processor.get_dev_examples(args.dev_file)
        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_eids = torch.tensor([f.eid for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_eids, all_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, num_workers=16)
        eval_label_ids = all_label_ids
        eval_eids = all_eids

    if args.do_train:
        train_examples = processor.get_train_examples(args.train_file)
        train_features = convert_examples_to_features(
                train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            model = BertForSequenceClassification.from_pretrained(
                args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_labels=num_labels)
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = BertAdam(
                optimizer_grouped_parameters, lr=lr,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps)

            start_time = time.time()
            global_step = 0
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(tqdm(train_batches)):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()

                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                     epoch, step + 1, len(train_batches),
                                     time.time() - start_time, tr_loss / nb_tr_steps))
                        save_model = False
                        if args.do_eval:
                            preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, eval_eids, num_labels, id2label)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            # logger.info("First 20 predictions:")
                            # for pred, label in zip(preds[:20], eval_label_ids.numpy()[:20]):
                            #     sign = u'\u2713' if pred == label else u'\u2718'
                            #     logger.info("pred = %s, label = %s %s" % (id2label[pred], id2label[label], sign))
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(result.keys()):
                                        writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_eval:
        if args.eval_test:
            eval_examples, raw_data = processor.get_test_examples(args.test_file)
            eval_features = convert_examples_to_features(
                eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_eids = torch.tensor([f.eid for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_eids, all_label_ids)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, num_workers=16)
            eval_label_ids = all_label_ids
            eval_eids = all_eids
        else:
            raw_data = None
        model = BertForSequenceClassification.from_pretrained(args.finetune_dir, num_labels=num_labels)
        model.to(device)
        preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, eval_eids, num_labels, id2label, raw_data=raw_data)
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for ex, pred in zip(eval_examples, preds):
                f.write("%s\t%s\n" % (ex.guid, id2label[pred]))
        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
            for key in sorted(result.keys()):
                f.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The train file for the task.")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="The dev file for the task.")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The test file for the task.")
    parser.add_argument("--finetune_dir", default=None, type=str, required=True,
                        help="The directory where the finetuned checkpoints stored.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--feature_mode", type=str, default="ner", choices=["text", "ner", "text_ner", "ner_text"])
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    main(args)
