import os
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from llm import LLM
import aug


def get_aug_sents(data_file, aug_keys=None):
    data = json.load(open(data_file))
    if aug_keys is None:
        re_stats = defaultdict(int)
        for sent in data:
            re_stats[sent['relation']] += 1
        aug_keys = set()
        for key, value in sorted(re_stats.items(), key=lambda x: x[1]):
            if value < 300:
                aug_keys.add(key)
    print(aug_keys)
    target_sents = [sent for sent in data if sent['relation'] in aug_keys]
    print(len(target_sents))
    return target_sents, aug_keys


def aug_sents(llm, aug_sents, post_fix='1', save_folder='augs'):
    template = """
    You are an editor who is very good at paraphrasing sentences. Your task is rewrite a given sentence well keeping the original entities.

    In a sentence, two entities are nested in the sentence in the format of [[ entity ]].
    Rewrite the given sentence using each given entity exactly once and do not introduce other entities.
    Nest the original entities in the same format in the rewrited sentence.
    You can change the content inside the entity.

    %s
    """

    encoded_sents = aug.get_encoded_sents(aug_sents)

    for relation, cur_encoded_sents in sorted(encoded_sents.items(), key=lambda x: len(x[1])):
        print('--------> processing', relation, len(cur_encoded_sents))
        file_name = f'{relation.replace("/", "--")}_{post_fix}.json'
        save_path = os.path.join(save_folder, file_name)
        if os.path.exists(save_path):
            print(f'skip {relation} due to {save_path} exists')
            continue
        rewrited_sents = []
        for encoded_sent, head_ent_type, tail_ent_type, head_ent, tail_ent in tqdm(cur_encoded_sents):
            message = template % encoded_sent
            rewrited_sent = aug.rewrite_sent(llm, message)
            if rewrited_sent:
                # print('*' * 30)
                # print(encoded_sent)
                # print(rewrited_sent)
                rewrited_sents.append((rewrited_sent, head_ent_type, tail_ent_type, head_ent, tail_ent))
        print(f'--------> {len(rewrited_sents)} sentences generated')
        print('*' * 50)
        with open(save_path, 'w') as af:
            json.dump(rewrited_sents, af)


if __name__ == '__main__':
    args = sys.argv[1:]
    device = int(args[0]) if len(args) else 0
    post_fix = f'{device}'
    print(args, '------- using post_fix', post_fix)
    llm = LLM(f'cuda:{device}')
    aug_keys = None
    for key in ['train', 'dev', 'test']:
        data_file = f'datasets/retacred/{key}.json'
        aug_folder = f'augs_{key}'
        print(f'-------> working on {key} data, and output to {aug_folder}')
        sents, aug_keys = get_aug_sents(data_file, aug_keys)
        aug_sents(llm, sents, post_fix, aug_folder)
