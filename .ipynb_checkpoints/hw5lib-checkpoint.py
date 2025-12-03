import os
import re
import math
import sqlite3
from collections import defaultdict

_WORD_RE = re.compile(r"[A-Za-z']+")

def _normalize(w: str) -> str:
    w = w.lower().strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    if w.endswith("s") and not (w.endswith("ss") or w.endswith("us")):
        w = w[:-1]

    if w.startswith("intellig"):          # intelligent, intelligence, intelligently
        return "intellig"

    if len(w) > 3:
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ic"):
            return w[:-2]
        if w.endswith("ical"):
            return w[:-4]
        if w.endswith("es"):
            return w[:-2]
        if w.endswith("e") and not w.endswith("re"):
            return w[:-1]
    return w

def _tokens(text: str):
    tokens = []
    for t in _WORD_RE.findall(text):
        w = _normalize(t)
        if not w:
            continue
        if w in _STOPWORDS:
            continue
        tokens.append(w)
    return tokens


# 1.1. Create database
def create_db():
    if os.path.exists("index.db"):
        os.remove("index.db")

    conn = sqlite3.connect("index.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS index_table (
            term TEXT,
            file TEXT,
            count INTEGER,
            PRIMARY KEY (term, file)
        )
    """)
    conn.commit()
    conn.close()

# 1.2. Index a single file
def index(filename):
    if not os.path.isfile(filename):
        return
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        tokens = _tokens(f.read())
    term_counts = defaultdict(int)
    for t in tokens:
        term_counts[t] += 1
    norm_path = filename.replace("\\", "/")  # normalize slashes
    for term, count in term_counts.items():
        cur.execute("""
            INSERT OR REPLACE INTO index_table (term, file, count)
            VALUES (?, ?, ?)
        """, (term, norm_path, count))
    conn.commit()
    conn.close()

# 1.3. Index all files in a directory
def index_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                index(os.path.join(root, file))

# 1.4. Find postings for a list of terms
def find(terms):
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    results = {}
    for raw in terms:
        term = _normalize(raw)
        rows = cur.execute(
            "SELECT file, count FROM index_table WHERE term = ?",
            (term,)
        ).fetchall()
        results[raw] = {file: cnt for file, cnt in rows}

    conn.close()
    return results

# 1.5. Search by query string
def search(query: str):
    terms = _tokens(query)
    postings = find(terms)

    # collect per-doc per-term counts
    per_doc = defaultdict(lambda: defaultdict(int))
    for term, docs in postings.items():
        for f, c in docs.items():
            per_doc[f][term] += c

    ranked = sorted(
        per_doc.items(),
        key=lambda item: (
            -sum(1 for cnt in item[1].values() if cnt > 0), 
            -sum(item[1].values()),                            
            item[0]                                            
        )
    )
    return [f for f, _ in ranked]





# hw4
# 1. Building virtual term documents
_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","had","he","her",
    "his","i","in","is","it","its","of","on","or","that","the","their","there","they","this",
    "to","was","were","will","with","you","your","we","our"
}

def index_context_vectors(k: int):
    k = max(0, int(k))
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    # table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS context_vectors (
            term TEXT,
            neighbor TEXT,
            score REAL,
            rank INTEGER,
            PRIMARY KEY (term, neighbor)
        )
    """)
    cur.execute("DELETE FROM context_vectors")

    # Load tf per (term,file)
    rows = cur.execute("SELECT term, file, count FROM index_table").fetchall()

    # term -> {file: tf}
    tf = defaultdict(dict)
    # file -> set(terms) for quick iteration
    file_terms = defaultdict(set)
    for t, f, c in rows:
        tf[t][f] = int(c)
        file_terms[f].add(t)

    # Total counts per term (∑_d tf_{t,d})
    total_tf = {t: sum(file_tf.values()) for t, file_tf in tf.items()}

    # Compute neighbors
    for t, tf_t in tf.items():
        if t in _STOPWORDS:
            continue

        co_counts = defaultdict(int)  # n_ij = ∑_d min(tf_{t,d}, tf_{u,d})
        for f, c_t in tf_t.items():
            # iterate all terms in same doc
            for u in file_terms[f]:
                if u == t or u in _STOPWORDS:
                    continue
                c_u = tf[u].get(f, 0)
                if c_u:
                    co_counts[u] += min(c_t, c_u)

        # Dice with token counts
        scored = []
        denom_t = total_tf.get(t, 0)
        if denom_t == 0:
            continue
        for u, n_ij in co_counts.items():
            denom_u = total_tf.get(u, 0)
            if denom_u == 0:
                continue
            dice = (2.0 * n_ij) / (denom_t + denom_u)
            scored.append((u, dice))

        scored.sort(key=lambda x: (-x[1], x[0]))
        for r, (u, s) in enumerate(scored[:k], start=1):
            cur.execute(
                "INSERT OR REPLACE INTO context_vectors(term, neighbor, score, rank) VALUES (?,?,?,?)",
                (t, u, float(s), r),
            )

    conn.commit()
    conn.close()


def get_context_vector(term: str):
    t = _normalize(term)
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT neighbor FROM context_vectors WHERE term=? ORDER BY rank ASC, neighbor ASC",
        (t,),
    ).fetchall()
    conn.close()
    return [n for (n,) in rows]


def suggest(query: str):
    q = [t for t in _tokens(query) if t and t not in _STOPWORDS]
    if not q:
        return []

    conn = sqlite3.connect("index.db")
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in q)

    # term, neighbor, rank for neighbors that are in the query set
    rows = cur.execute(
        f"""SELECT term, neighbor, rank
            FROM context_vectors
            WHERE neighbor IN ({placeholders})""",
        tuple(q),
    ).fetchall()
    conn.close()

    ranks_by_key = defaultdict(dict)
    for term, neighbor, rank in rows:
        ranks_by_key[term][neighbor] = int(rank)

    candidates = []
    for key, seen in ranks_by_key.items():
        if all(t in seen for t in q):  # covers all query terms
            worst = max(seen[t] for t in q)
            total = sum(seen[t] for t in q)
            candidates.append((key, worst, total))

    candidates.sort(key=lambda x: (x[1], x[2], x[0]))
    return [k for k, _, _ in candidates]



# 3. Search Results
def highlight_keywords(sentence, positions, fmt="**{}**"):
    if not positions:
        return sentence
    parts = []
    last = 0
    for (s, e) in sorted(positions):
        parts.append(sentence[last:s])
        parts.append(fmt.format(sentence[s:e]))
        last = e
    parts.append(sentence[last:])
    return "".join(parts)

def score_sentence(keypairs, sentence):
    text = sentence
    lower = text.lower()

    # tokens with char positions
    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text)]

    positions = []
    score = 0
    seen = set()  # avoid duplicate spans

    for tok, s, e in tokens:
        t = tok.lower()
        matched = False
        for w, st in keypairs:
            w = w.lower()
            st = st.lower()

            if t == w:                 # exact word
                val = 2
                matched = True
            elif st and t.startswith(st):  # stem/prefix (handles plurals, -ing, etc.)
                val = 1
                matched = True
            else:
                continue

            if matched:
                if (s, e) not in seen:
                    positions.append((s, e))
                    seen.add((s, e))
                    score += val
                break  # don’t double count this token for multiple keypairs

    positions.sort()

    # proximity bonus: +1 for each pair within 50 chars
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[j][0] - positions[i][0] <= 50:
                score += 1

    return score, positions

def snippet(query, filename, max_length=250):
    keypairs = [(w, _normalize(w)) for w in _tokens(query)]
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # simple sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)

    scored = []
    for s in sentences:
        sc, pos = score_sentence(keypairs, s)
        if sc > 0:
            scored.append((sc, s, pos))

    if not scored:
        return ""

    # highest score first
    scored.sort(key=lambda x: -x[0])

    out = []
    total = 0
    used_indexes = set()

    # take best sentences while respecting max_length; keep their original order
    chosen = []
    for sc, s, pos in scored:
        if total + len(s) <= max_length:
            chosen.append((s, pos))
            total += len(s)

    # maintain source order by their first occurrence in the doc
    ordered = []
    start = 0
    for s, _ in chosen:
        idx = text.find(s, start)
        ordered.append((idx, s))
        start = idx + len(s)
    ordered.sort()

    last_idx = None
    for _, s in ordered:
        _, pos = score_sentence(keypairs, s)  # recompute positions in this exact sentence
        piece = highlight_keywords(s, pos, "<b>{}</b>")
        if out and last_idx is not None:
            out.append(" ... ")
        out.append(piece)
        last_idx = 1

    return "".join(out).strip()





# hw5
def add_docs(directory):
    index_dir(directory)

# 1.1
def index_vector_model():
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_vectors (
            file   TEXT,
            term   TEXT,
            weight REAL,
            PRIMARY KEY (file, term)
        )
    """)
    cur.execute("DELETE FROM doc_vectors")

    doc_rows = cur.execute("SELECT DISTINCT file FROM index_table").fetchall()
    docs = [f for (f,) in doc_rows]
    N = len(docs)
    if N == 0:
        conn.commit()
        conn.close()
        return

    df_rows = cur.execute("""
        SELECT term, COUNT(DISTINCT file)
        FROM index_table
        GROUP BY term
    """).fetchall()
    df = {term: int(cnt) for term, cnt in df_rows}

    idf = {}
    for term, n_k in df.items():
        if n_k > 0:
            idf[term] = math.log(N / n_k)
        else:
            idf[term] = 0.0

    for fname in docs:
        rows = cur.execute(
            "SELECT term, count FROM index_table WHERE file = ?",
            (fname,)
        ).fetchall()

        weights = {}
        for term, c in rows:
            c = int(c)
            if c <= 0:
                continue
            # TF and IDF as in the equation
            tf = math.log(c) + 1.0
            w  = tf * idf.get(term, 0.0)
            if w != 0.0:
                weights[term] = w

        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0:
            continue

        for term, w in weights.items():
            cur.execute(
                "INSERT OR REPLACE INTO doc_vectors(file, term, weight) VALUES (?,?,?)",
                (fname, term, float(w / norm)),
            )

    conn.commit()
    conn.close()
    


# 1.2
def get_document_vector(filename: str):
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT term, weight FROM doc_vectors WHERE file = ?",
        (filename,),
    ).fetchall()
    conn.close()

    first_pos = {}
    try:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        toks = _tokens(text)
        for i, t in enumerate(toks):
            if t not in first_pos:
                first_pos[t] = i
    except FileNotFoundError:
        first_pos = {}

    rows.sort(
        key=lambda x: (
            -x[1],
            first_pos.get(x[0], 10**9),
            x[0],
        )
    )
    return rows



# 1.3
def rank_document_model(query: str):
    toks = _tokens(query)
    if not toks:
        return []

    q_tf = defaultdict(int)
    for t in toks:
        q_tf[t] += 1

    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    rows = cur.execute("SELECT DISTINCT file FROM index_table").fetchall()
    N = len(rows)
    if N == 0:
        conn.close()
        return []

    placeholders = ",".join("?" for _ in q_tf)
    df_rows = cur.execute(
        f"""
        SELECT term, COUNT(DISTINCT file)
        FROM index_table
        WHERE term IN ({placeholders})
        GROUP BY term
        """,
        tuple(q_tf.keys()),
    ).fetchall()
    df = {term: cnt for term, cnt in df_rows}

    q_raw = {}
    for term, f_qk in q_tf.items():
        if term not in df or df[term] == 0:
            continue
        tf = math.log(f_qk) + 1.0
        idf = math.log(N / df[term])
        w = tf * idf
        q_raw[term] = w

    if not q_raw:
        conn.close()
        return []

    q_norm = math.sqrt(sum(w * w for w in q_raw.values()))
    if q_norm == 0:
        conn.close()
        return []

    q_vec = {term: w / q_norm for term, w in q_raw.items()}

    doc_vecs = defaultdict(dict)
    for fname, term, weight in cur.execute("SELECT file, term, weight FROM doc_vectors"):
        doc_vecs[fname][term] = float(weight)

    conn.close()

    scores = []
    for fname, dvec in doc_vecs.items():
        score = 0.0
        for term, q_w in q_vec.items():
            d_w = dvec.get(term)
            if d_w:
                score += q_w * d_w
        if score > 0:
            scores.append((score, fname))

    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores



# 1.4
def query_document_model(query: str):
    ranked = rank_document_model(query)
    return [fname for (_, fname) in ranked]



# 2.1
def index_query_likelihood(mu=2000):
    
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_lm (
            term TEXT,
            file TEXT,
            prob REAL,
            PRIMARY KEY (term, file)
        )
    """)
    cur.execute("DELETE FROM doc_lm")

    rows = cur.execute("SELECT term, file, count FROM index_table").fetchall()

    tf = defaultdict(lambda: defaultdict(int))

    doc_len = defaultdict(int)
    collection_count = defaultdict(int)

    for term, fname, cnt in rows:
        cnt = int(cnt)
        tf[term][fname] = cnt
        doc_len[fname] += cnt
        collection_count[term] += cnt

    C = sum(collection_count.values())

    collection_prob = {t: collection_count[t] / C for t in collection_count}

    for term in collection_count:
        bg = collection_prob[term]
        for fname in doc_len:
            f = tf[term].get(fname, 0)
            D = doc_len[fname]
            p = (f + mu * bg) / (D + mu)
            cur.execute(
                "INSERT OR REPLACE INTO doc_lm(term, file, prob) VALUES (?,?,?)",
                (term, fname, float(p)),
            )

    conn.commit()
    conn.close()



# 2.2
def get_document_likelihood(filename: str):
    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT term, prob FROM doc_lm WHERE file = ?",
        (filename,),
    ).fetchall()

    conn.close()

    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows



# 2.3
def rank_query_likelihood(query: str):
    q_tokens = _tokens(query)
    if not q_tokens:
        return []

    conn = sqlite3.connect("index.db")
    cur = conn.cursor()

    docs = [f for (f,) in cur.execute("SELECT DISTINCT file FROM doc_lm")]

    lm = defaultdict(dict)
    for term, fname, p in cur.execute("SELECT term, file, prob FROM doc_lm"):
        lm[fname][term] = float(p)

    conn.close()

    scores = []
    for fname in docs:
        score = 0.0
        ok = True
        for t in q_tokens:
            if t not in lm[fname]:
                ok = False
                break
            p = lm[fname][t]
            score += math.log(p)
        if ok:
            scores.append((score, fname))

    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores



# 2.4 
def query_likelihood(query: str):
    ranked = rank_query_likelihood(query)
    return [fname for (_, fname) in ranked]