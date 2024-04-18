import os
import json
from collections import defaultdict
from glob import glob
from tqdm import tqdm
from rapidfuzz import fuzz


def rewrite_sent(llm, message, max_iter=10):
    is_valid = False
    cur_iter = 0
    while not is_valid and cur_iter < max_iter:
        response = llm.chat(message)
        new_tokens = decode(response)
        is_valid, error = check_valid(new_tokens)
        cur_iter += 1
    if not is_valid:
        return None
    return new_tokens


def decode(content):
    pre_words = content.split()
    while True:
        changed = False
        words = []
        for word in pre_words:
            if len(word) == 0:
                continue
            if word in ['[[', ']]']:
                words.append(word)
            elif '[[' in word:
                idx = word.index('[[')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append('[[')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                changed = True
            elif ']]' in word:
                idx = word.index(']]')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append(']]')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                changed = True
            elif not word[-1].isalnum():
                if len(word[:-1]):
                    words.append(word[:-1])
                words.append(word[-1])
            else:
                words.append(word)
        if not changed:
            break
        pre_words = words
    return words


def check_valid(words):
    if '\n' in words:
        return False, 'multiple lines'
    if words.count('[[') != 2:
        return False, 'incorrect num of [[: %s' % words.count('[[')
    if words.count(']]') != 2:
        return False, 'incorrect num of ]]: %s' % words.count(']]')
    head_start = words.index('[[')
    head_end = words.index(']]')
    tail_start = words.index('[[', head_start+1)
    tail_end = words.index(']]', head_end+1)
    if set(range(head_start, head_end+2)) & set(range(tail_start, tail_end+2)):
        return False, 'overlap entities'
    return True, 'Good content'


def encode_sent(sent):
    tokens = list(sent['token'])
    pairs = [(sent['subj_start'], sent['subj_end']+1, '[[', ']]'), (sent['obj_start'], sent['obj_end']+1, '[[', ']]')]
    pairs.sort()
    poss = set()
    for start, end, _, _ in pairs:
        if set(range(start, end)) & poss:
            print('------> overlapping entities')
            return None
        poss.update(range(start, end))
    for idx, (start, end, head_token, tail_token) in enumerate(pairs):
        tokens.insert(start+2*idx, head_token)
        tokens.insert(end+2*idx+1, tail_token)
    return ' '.join(tokens)


def get_encoded_sents(sents):
    encoded_sents = defaultdict(list)
    for sent in sents:
        encoded_sent = encode_sent(sent)
        if encoded_sent:
            head_ent_type, tail_ent_type = sent['subj_type'], sent['obj_type']
            head_ent = sent['token'][sent['subj_start']: sent['subj_end']+1]
            tail_ent = sent['token'][sent['obj_start']: sent['obj_end']+1]
            sent_info = {
                'encoded_sent': encoded_sent, 'head_ent_type': head_ent_type, 'tail_ent_type': tail_ent_type,
                'head_ent': head_ent, 'tail_ent': tail_ent, 'id': sent['id']}
            encoded_sents[sent['relation']].append(sent_info)
    return encoded_sents


def transform_aug_file(aug_file):
    relation = os.path.basename(aug_file).rsplit('_', 1)[0].replace('--', '/')
    cur_data = json.load(open(aug_file))
    new_cur_data = []
    for idx, sent_info in enumerate(cur_data):
        words = sent_info['rewrited_sent']
        head_words, tail_words = sent_info['head_ent'], sent_info['tail_ent']
        head_start = words.index('[[') - 0
        head_end = words.index(']]') - 1
        tail_start = words.index('[[', head_start+1) - 1
        tail_end = words.index(']]', head_end+2) - 2
        words.remove('[[')
        words.remove(']]')

        if head_start >= head_end or tail_start >= tail_end:
            continue

        assert head_end < len(words)
        assert tail_end < len(words)
        assert head_start >= 0
        assert tail_start >= 0
        assert head_start < head_end
        assert tail_start < tail_end

        cur_head_words = words[head_start: head_end]
        head_head_ratio = fuzz.ratio(' '.join(cur_head_words), ' '.join(head_words))
        head_tail_ratio = fuzz.ratio(' '.join(cur_head_words), ' '.join(tail_words))
        if head_head_ratio < head_tail_ratio:
            head_start, head_end, tail_start, tail_end = tail_start, tail_end, head_start, head_end

        sent = {
            'id': sent_info['id'], 'token': words, 'subj_start': head_start, 'subj_end': head_end-1,
            'obj_start': tail_start, 'obj_end': tail_end-1, 'relation': relation,
            'subj_type': sent_info['head_ent_type'], 'obj_type': sent_info['tail_ent_type']}
        new_cur_data.append(sent)
    return new_cur_data


def get_auged_sents(folder):
    aug_files = glob(f'{folder}/*.json')
    sents = []
    for aug_file in aug_files:
        sents.extend(transform_aug_file(aug_file))
    return sents


def rewrite_sents(llm, template, encoded_sents, post_fix):
    rewrited_sents = []
    for sent_info in tqdm(encoded_sents):
        message = template % sent_info['encoded_sent']
        rewrited_sent = rewrite_sent(llm, message)
        if rewrited_sent:
            # print('*' * 30)
            # print(encoded_sent)
            # print(rewrited_sent)
            sent_info['rewrited_sent'] = rewrited_sent
            sent_info['id'] = f'{sent_info["id"]}_{post_fix}'
            rewrited_sents.append(sent_info)
    return rewrited_sents
