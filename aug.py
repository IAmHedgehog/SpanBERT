from collections import defaultdict


def rewrite_sent(llm, message, max_iter=10):
    # message = template % sent
    is_valid = False
    cur_iter = 0
    while not is_valid and cur_iter < max_iter:
        response = llm.chat(message)
        new_tokens = decode(response)
        is_valid, error = check_valid(new_tokens)
        cur_iter += 1
    if not is_valid:
        # print('=====not valid afater %s tries' % max_iter)
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
            if word in ['[[', ']]', '<<', '>>']:
                words.append(word)
            elif '[[' in word:
                idx = word.index('[[')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append('[[')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                # changed = True
            elif '<<' in word:
                idx = word.index('<<')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append('<<')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                # changed = True
            elif ']]' in word:
                idx = word.index(']]')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append(']]')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                # changed = True
            elif '>>' in word:
                idx = word.index('>>')
                if len(word[:idx]):
                    words.append(word[:idx])
                words.append('>>')
                if len(word[idx+2:]):
                    words.append(word[idx+2:])
                # changed = True
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
    # if words.count('<<') != 0:
    #     return False, 'incorrect num of <<: %s' % words.count('<<')
    # if words.count('>>') != 0:
    #     return False, 'incorrect num of >>: %s' % words.count('>>')
    head_start = words.index('[[')
    head_end = words.index(']]')
    tail_start = words.index('[[', head_start+1)
    tail_end = words.index(']]', head_end+1)
    if set(range(head_start, head_end+2)) & set(range(tail_start, tail_end+2)):
        return False, 'overlap entities'
    return True, 'Good content'


def encode_sent(sent):
    tokens = list(sent['token'])
    # pairs = [(sent['subj_start'], sent['subj_end']+1, '[[', ']]'), (sent['obj_start'], sent['obj_end']+1, '<<', '>>')]
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
        # print(' '.join(sent['tokens']))
        encoded_sent = encode_sent(sent)
        if encoded_sent:
            head_ent_type, tail_ent_type = sent['subj_type'], sent['obj_type']
            head_ent = sent['token'][sent['subj_start']: sent['subj_end']+1]
            tail_ent = sent['token'][sent['obj_start']: sent['obj_end']+1]
            encoded_sents[sent['relation']].append((encoded_sent, head_ent_type, tail_ent_type, head_ent, tail_ent))
    return encoded_sents
