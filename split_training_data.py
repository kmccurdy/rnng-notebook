#python3

import os, random, math, re

indir = "data/treebank/combined" ## directory with PTB files
pct_train = .7 # percent of files to be used for training set
pct_dev = .15 # percent of files to be used for development set

outfile_train = "data/train.ptb"
outfile_dev = "data/dev.ptb"
outfile_test = "data/test.ptb"
outfile_sources = "data/source_files.tsv" # keep a record of which PTB files end up in which dataset

# regular expression for cleaning nonterminal symbols
# i.e. PP-LOC => PP, NP-SBJ-1 => NP, VP2 => VP, etc.
NT_re = re.compile("\s*\((NP|VP|PP|ADJP|ADVP|WHNP|WHADVP|WHADJP|WHPP|SBAR|FRAG|UCP|NAC|QP|NX|LST|CONJP|X|S)[^\(]*")

# regular expression to identify & remove syntactic traces
trace_re = re.compile("\(-NONE-[^\)]*\)")
trace_re_parent = re.compile("\([A-Z]+ ? ?\(-NONE-[^\)]*\) ?\)")

def NT_clean(line):
  if NT_re.match(line):
    NT = NT_re.match(line)
    line = line.replace(NT.group(0), "("+NT.group(1)+" ")
  return line

def trace_clean(line):
  while trace_re_parent.search(line):
    line = line.replace(trace_re_parent.search(line).group(0), "")
  while trace_re.search(line):
    line = line.replace(trace_re.search(line).group(0), "")
  return line 

if __name__ == '__main__':
  files = os.listdir(indir)
  print(len(files))
  random.shuffle(files)
  n_train = math.floor(pct_train * len(files))
  n_dev = math.floor(pct_dev * len(files)) + n_train
  f_train = open(outfile_train, 'w')
  f_dev = open(outfile_dev, 'w')
  f_test = open(outfile_test, 'w')
  f_sources = open(outfile_sources, 'w')
  f_sources.write('\t'.join(["type", "file", "n_sentences", "up_to_line"])+"\n")
  line_counter = 0
  last_source = ""
  for i, infile in enumerate(files):
    if infile.endswith(".mrg"):
      with open(indir + "/" + infile, 'r') as inf:
        if i <= n_train:
          source = "train"
          writefile = f_train
        elif i <= n_dev:
          source = "dev"
          writefile = f_dev
        else:
          source = "test"
          writefile = f_test
        tree = ""
        sentence_counter = 0
        if last_source != source:
          line_counter = 0
        for line in inf:
          if line.startswith("( (") or line.startswith("(("):
            line = NT_clean(line[1:].strip()) # trim opening empty bracket
            tree = tree.rstrip()[:-1] # trim closing empty bracket
            tree = trace_clean(tree)
            writefile.write(tree)
            tree = "\n" + line + " " # newline, then start next tree
            if line_counter == 0: # if starting a new file:
              tree = tree.lstrip() # remove initial newline
            sentence_counter += 1 
            line_counter += 1
          else:
            if line.strip():
              line = line.strip()
              tree += NT_clean(line) + ' '
        last_source = source
        f_sources.write("\t".join([source, infile, str(sentence_counter), str(line_counter)])+"\n")
  for f in [f_train, f_dev, f_test, f_sources]:
    f.close()
