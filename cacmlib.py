import os
from pathlib import Path
import regex

# Change this directory if needed
DIR_DATASETS="../../datasets"

cacm_test_queries = [ 
    ('Parallel languages languages for parallel computation', 10),
    ('SETL Very High Level Languages', 11),
    ('portable operating systems', 12),
    ('code optimization for space efficiency', 13),
    ('parallel algorithms', 19),
    ('applied stochastic processes', 24),
    ('parallel processes in information retrieval', 50),
]

def cacm_query(qid):
    with open(f"{DIR_DATASETS}/cacm/cacm-query.txt") as f:
        for line in f:
            m = regex.regex.match("^\\[(\\d+)\\] +(.*)$", line)
            if not m: continue
            if qid == int(m[1]): return m[2]
    return None

def cacm_contents(id):
    try:
        with open(f"{DIR_DATASETS}/cacm/CACM-{id:04d}.html") as f:
            return f.read()
    except:
        return None

def cacm_rel_key(query_id, k=None, rel_file=f"{DIR_DATASETS}/cacm/cacm-rel.txt"):
    results = []
    with open(rel_file) as file:
        for line in file:
            (qid, _, name, _) = line.split()
            if int(qid) == query_id:
                (prefix, num) = name.split('-')
                results.append(f"{prefix}-{int(num):04d}")
    return results[:k]

def cacm_contents_search(query, fn):
    results = {}
    for f in os.listdir("{DIR_DATASETS}/cacm"):
        filename = f"{DIR_DATASETS}/cacm/{f}"
        if os.path.isdir(filename): continue
        with open(filename) as file:
            contents = file.read()
            results[Path(f).stem] = fn(query, contents)
    return results

# Normalize the results of a standard search function
def cacm_search(query, fn):
    results = fn(query)
    return [ os.path.splitext(os.path.basename(filename))[0] for filename in results ]

def f_score(docs, rels):
    if not rels: return (0, 0, 0)
    reldocs = sum([ 1 if f in rels else 0 for f in docs ])
    if reldocs == 0: return (0, 0, 0)
    recall =  reldocs / len(rels)
    precision = reldocs / len(docs)
    f1 = 2 * precision * recall / (precision + recall)
    return (recall, precision, f1)

def precision_at(rank, docs, rels):
    reldocs = sum([ 1 if f in rels else 0 for f in docs[:rank]])
    precision = reldocs / rank
    return precision

def precisions(docs, rels):
    ps = []
    for i in range(0,len(docs)):
        if docs[i] in rels:
            ps.append(precision_at(i+1, docs, rels))
    return ps

def avg_precision(docs, rels):
    ps = precisions(docs, rels)
    return sum(ps) / len(ps) if len(ps) > 0 else 0

def reciprocal_rank(docs, rels):
    for i in range(len(docs)):
        if docs[i] in rels: return 1/(i+1)
    return 0

def search(query, search_fn):
    results = cacm_search(query, search_fn)
    sorted = list(results.keys())
    sorted.sort(key=lambda x: results[x])
    return sorted

def eval(search_fn):
    scores = {}
    queries = 0
    rr = 0
    avg_p = 0
    with open(f"{DIR_DATASETS}/cacm/cacm-query.txt") as f:
        for line in f:
            m = regex.regex.match("^\\[(\\d+)\\] +(.*)$", line)
            if not m: continue
            queries += 1
            qid = int(m[1])
            query = m[2]
            docs = cacm_search(query, search_fn)
            rels = cacm_rel_key(qid)
            rr += reciprocal_rank(docs, rels)
            avg_p += avg_precision(docs, rels)
    return (rr/queries), (avg_p/queries)

def print_score(score, baseline = None):
    delta = f" : {((score[0] - baseline[0]) / baseline[0]) * 100:.2f}%" if baseline else ""
    print(f"MRR : {score[0]}{delta}")
    delta = f" : {((score[1] - baseline[1]) / baseline[1]) * 100:.2f}%" if baseline else ""
    print(f"AVGP: {score[1]}{delta}")
    
    