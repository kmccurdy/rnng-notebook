from operator import itemgetter
from itertools import count, chain
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re

# Adapted from https://github.com/neubig/nn4nlp-code/blob/master/12-transitionparsing/stack_lstm.py
# and https://github.com/clab/rnng/blob/master/nt-parser/nt-parser-gen.cc

train_file = "data/train.oracle"
dev_file = "data/dev.oracle"
test_file = "data/test.oracle"
cluster_file = "data/bllip_clusters"

class Vocab:
  def __init__(self, w2i):
    self.w2i = dict(w2i)
    self.i2w = {i:w for w,i in w2i.items()}

  @classmethod
  def from_list(cls, words):
    w2i = {}
    idx = 0
    for word in words:
      w2i[word] = idx
      idx += 1
    return Vocab(w2i)

  @classmethod
  def from_file(cls, vocab_fname):
    words = []
    with open(vocab_fname) as fh:
      for line in fh:
        line.strip()
        cluster, word, count = line.split("\t")
        words.append(word)
    return Vocab.from_list(words)  

  def merge_vocab(self, dic):
    self.w2i = Vocab(self.w2i.keys() + dic.keys()).w2i
    self.i2w = {i:w for w,i in self.w2i.items()}

  def size(self): return len(self.w2i.keys())

class TransitionParser:
  def __init__(self, model, cluster_filepath, vocab_acts, WORD_DIM=50, LSTM_DIM=256, ACTION_DIM=16):
    self.vocab = Vocab.from_file(cluster_filepath)
    #self.vocab.w2i["UNK"] = self.vocab.size() + 1 # add "UNK" to vocabulary... nope =/
    self.vocab_acts = vocab_acts
    self.vocab_NTs = Vocab.from_list(get_NTs(vocab_acts.w2i.keys()))
    self.act_NT_map = dict([[vocab_acts.w2i[x], self.vocab_NTs.w2i[x[3:-1]]] 
                            for x in vocab_acts.w2i if x.startswith("NT")])

    # parameters
    self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM*2))
    self.pb_comp = model.add_parameters((LSTM_DIM, ))
    self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM ))
    self.pb_s2h = model.add_parameters((LSTM_DIM, ))
    self.pW_act = model.add_parameters((vocab_acts.size(), LSTM_DIM))
    self.pb_act = model.add_parameters((vocab_acts.size(), ))
    self.pempty_stack_emb = model.add_parameters((LSTM_DIM,)) # empty stack embedding /root guard

    # layers, in-dim, out-dim, model
    self.stackRNN = dy.CoupledLSTMBuilder(2, LSTM_DIM, LSTM_DIM, model)
    self.comp_LSTM_fwd = dy.CoupledLSTMBuilder(2, LSTM_DIM, LSTM_DIM, model)
    self.comp_LSTM_rev = dy.CoupledLSTMBuilder(2, LSTM_DIM, LSTM_DIM, model)
    self.cfsm = dy.ClassFactoredSoftmaxBuilder(LSTM_DIM, cluster_filepath, self.vocab.w2i, model);

    # lookup params
    self.WORDS_LOOKUP = model.add_lookup_parameters((self.vocab.size(), LSTM_DIM))
    self.ACT_LOOKUP = model.add_lookup_parameters((self.vocab_acts.size(), ACTION_DIM))
    self.NT_LOOKUP = model.add_lookup_parameters((self.vocab_NTs.size(), LSTM_DIM))

    self.pc = model

  def gen_setup(self, dropout=None):
    dy.renew_cg()
    stack = []
    stack.append((self.stackRNN.initial_state().add_input(dy.parameter(self.pempty_stack_emb)), 
      "<ROOT GUARD>")) # stack holds tuples: RNN state, string rep
    W_comp = dy.parameter(self.pW_comp)
    b_comp = dy.parameter(self.pb_comp)
    W_s2h = dy.parameter(self.pW_s2h)
    b_s2h = dy.parameter(self.pb_s2h)
    W_act = dy.parameter(self.pW_act)
    b_act = dy.parameter(self.pb_act)
    if dropout:
      self.stackRNN.set_dropout(dropout)
      self.comp_LSTM_fwd.set_dropout(dropout)
      self.comp_LSTM_rev.set_dropout(dropout)
    else:
      self.stackRNN.disable_dropout()
      self.comp_LSTM_fwd.disable_dropout()
      self.comp_LSTM_rev.disable_dropout()
    return [stack, # initial stack state,
            {"comp": (b_comp, W_comp), # bias and weight params for composition
              "s2h": (b_s2h, W_s2h), # bias and weight params for parser state
              "act": (b_act, W_act)}] # bias and weight params for predicting actions 

  def get_valid_actions(self, stack, open_nts, open_nt_ceil=100):
    # based on stack state, get valid actions
    valid_actions = []
    n_open_nts = len(open_nts)
    if n_open_nts < open_nt_ceil:  
      valid_actions += [v for k, v in self.vocab_acts.w2i.items() if k.startswith("NT")]
    if n_open_nts >= 1 and len(stack) > 1:
      valid_actions += [self.vocab_acts.w2i["SHIFT"]]
    if n_open_nts >= 1  and len(stack) > 1 \
        and len(stack) - 1 > open_nts[-1]:  # top element on stack can't be open NT
      valid_actions += [self.vocab_acts.w2i["REDUCE"]]
    return valid_actions

  def predict_action(self, stack, params, valid_actions, dropout=None):
    stack_embedding = stack[-1][0].output() 
    if dropout:
      stack_embedding = dy.dropout(stack_embedding, dropout)
    parser_state = dy.rectify(dy.affine_transform([params["s2h"][0], params["s2h"][1], stack_embedding]))
    logits = dy.affine_transform([params["act"][0], params["act"][1], parser_state])
    log_probs = dy.log_softmax(logits, valid_actions)
    return log_probs 

  def get_action(self, stack, params, valid_actions, n_actions, train_acts=None, dropout=None):
    action = valid_actions[0]
    loss = None
    if len(valid_actions) > 1:
      log_probs = self.predict_action(stack, params, valid_actions, dropout)
      if train_acts:
        try:
          action = self.vocab_acts.w2i[train_acts[n_actions]]
        except IndexError:
          raise Exception("Correct action list exhausted, but not in final parser state.")
        loss = dy.pick(log_probs, action)
      else:
        action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
    return action, loss

  def do_action(self, stack, action, params, open_nts, n_terms, train_sent=None, dropout=None):
    if action in self.vocab_acts.i2w:
      act = action
      action = self.vocab_acts.i2w[action]
    else:
      act = self.vocab_acts.w2i[action]
    word, nt_index, loss = "", 0, None
    if action == "SHIFT":
      if train_sent:
        try:
          word = train_sent[n_terms]
        except IndexError:
          raise Exception("Generated more terms than found in training sentence")
        if word not in self.vocab.w2i: ### for now, treat clusters as vocab
          if word.lower() in self.vocab.w2i:
            word = word.lower()
          else:
            #word = "UNK" ### - all words not in cluster file get UNKified - OR...
            word = random.sample(self.vocab.w2i.keys(), 1)[0] # ...replaced w/rando in-vocab wd
            # TODO: ^ this is not at all optimal, figure out how to fix
        loss = -self.cfsm.neg_log_softmax(stack[-1][0].output(), self.vocab.w2i[word])
      else:
        word = self.vocab.i2w[self.cfsm.sample(stack[-1][0].output())]
      word_embedding = self.WORDS_LOOKUP[self.vocab.w2i[word]]
      stack.append((stack[-1][0].add_input(word_embedding), word))
    elif action == "REDUCE":
      children = []
      last_nt_idx = open_nts.pop()
      while len(stack) > last_nt_idx + 1:
        children.append(stack.pop()) 
      children.reverse()
      last_nt = stack.pop()
      fwd = self.comp_LSTM_fwd.initial_state().add_input(last_nt[0].output())
      rev = self.comp_LSTM_rev.initial_state().add_input(last_nt[0].output())
      for i, child in enumerate(children):
        fwd.add_input(child[0].output())
        rev.add_input(children[len(children) - i - 1][0].output())
      cfwd = dy.dropout(fwd.output(), dropout) if dropout else fwd.output()
      crev = dy.dropout(rev.output(), dropout) if dropout else rev.output()
      bidir_rep = dy.concatenate([cfwd, crev])
      composed = dy.rectify(dy.affine_transform([params["comp"][0], 
                                                params["comp"][1], bidir_rep]))
      comp_str = last_nt[1] + " " + " ".join([child[1] for child in children]) + ")"
      stack.append((stack[-1][0].add_input(composed), comp_str))
    else: # open nonterminal
      NT = self.act_NT_map[act]
      nt_embedding = self.NT_LOOKUP[NT]
      stack.append((stack[-1][0].add_input(nt_embedding), "("+self.vocab_NTs.i2w[NT]))
      nt_index = len(stack) - 1
    return word, nt_index, loss

  def generate(self, train_sent=None, train_acts=None, dropout=None, nt_ceil=100):
    stack, params = self.gen_setup(dropout)
    terms, actions, open_nts, losses = [], [], [], []
    while len(terms) == 0 or len(stack) > 2 :
      valid_actions = self.get_valid_actions(stack, open_nts, nt_ceil)
      action, loss = self.get_action(stack, params, valid_actions, len(actions), train_acts, dropout)
      losses.append(loss) if loss else loss
      actions.append(action)
      term, nt_idx, loss = self.do_action(stack, action, params, open_nts, len(terms), train_sent, dropout)
      losses.append(loss) if loss else loss
      terms.append(term) if term else term
      open_nts.append(nt_idx) if nt_idx else nt_idx # open NT index is never 0
      if not train_sent:
        if len(terms) % 5 == 0:
          print(" ".join(terms))
    final_tree = stack[1][1]
    return final_tree, -dy.esum(losses) if losses else None

  def train(self, corpus, trainer, dev=None, dropout=.3, epochs=3, max_iter=None):
    i = 0
    corpus.sort(key=lambda x: len(x[1])) 
    order = list(range(len(corpus)))
    for epoch in range(epochs):
      random.shuffle(order)
      shuffled_corpus = [corpus[i] for i in order]
      words = 0
      total_loss = 0.0
      for (_, s,a) in shuffled_corpus:
        result, loss = self.generate(s, a, dropout)
        words += len(s)
        if loss is not None:
          total_loss += loss.scalar_value()
          loss.backward()
          trainer.update()
        e = float(i) / len(corpus)
        if i % 50 == 0:
          print('epoch {}: per-word loss: {}'.format(e, total_loss / words))
          if e > 1:
            result, loss = self.generate()
            print("   {}".format(result))
          words = 0
          total_loss = 0.0
        if i % 500 == 0 and dev:
          dev_words = 0
          dev_loss = 0.0
          for (_, ds, da) in dev:
            result, loss = self.generate(ds, da)
            dev_words += len(ds)
            if loss is not None:
              dev_loss += loss.scalar_value()
          print('[validation] epoch {}: per-word loss: {}'.format(e, dev_loss / dev_words))
        i += 1
        if max_iter:
          if i >= max_iter:
            break
      if max_iter:
        if i >= max_iter:
          break

def read_oracle(fname, gen=True):
  sent_idx = 1 if gen else 4 # using non-UNKified sentences
  act_idx = 3 if gen else 5
  with open(fname) as fh:
    sent_ctr = 0
    tree, sent, acts = "", [], []
    for line in fh:
      sent_ctr += 1
      line = line.strip()
      if line.startswith("#"):
        sent_ctr = 0
        if tree:
          yield tree, sent, acts
        tree, sent, acts = line, [], []
      if sent_ctr == sent_idx:
        sent = line.split()
      if sent_ctr >= act_idx:
        if line:
          acts.append(line)

def load_data(tr=train_file, d=dev_file, ts=test_file):
  train, dev, test = [], [], []
  if tr:
    train = list(read_oracle(tr))
  if d:
    dev = list(read_oracle(d))
  if ts:
    test = list(read_oracle(ts))
  return train, dev, test

def create_vocab(all_terms):
  vocab = list(set(list(chain(*all_terms))))
  return Vocab.from_list(vocab)

def get_NTs(actions):
  NTs = []
  for act in actions:
    if act.startswith("NT"):
      NTs.append(act[3:-1])
  return NTs

WORD_DIM = 50
LSTM_DIM = 256 # same for input and hidden layers
ACTION_DIM = 16 # dimension for action embedding

def main(tr=train_file, d=dev_file, ts=test_file):
  train, dev, test = load_data(tr, d, ts)
  # for the time being...
  train += test
  vocab_acts = create_vocab([x[2] for x in train])
  model = dy.ParameterCollection()
  tp = TransitionParser(model, cluster_file, vocab_acts)
  return model, tp, train, dev
