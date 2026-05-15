from nltk.corpus import wordnet
import json
import math


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def load_json(file_path):
    file_in = open(file_path, "r")
    data = json.load(file_in)
    file_in.close()
    return data


def save_json(data_to_save, target_path):
    file_out = open(target_path, "w")
    json.dump(data_to_save, file_out, indent=4)
    file_out.close()
    print("saved at:", target_path)


def get_relevant_docs(qrels_list):
    dict_rel = {}

    for i in range(len(qrels_list)):
        curr = qrels_list[i]

        q_id = int(curr["query_num"])
        d_id = int(curr["id"])
        pos = curr["position"]

        if q_id not in dict_rel:
            dict_rel[q_id] = []

        if pos == 1 or pos == 2 or pos == 3 or pos == 4:
            found = False
            for prev in dict_rel[q_id]:
                if prev == d_id:
                    found = True

            if found == False:
                dict_rel[q_id].append(d_id)

    return dict_rel


def preprocess_text(
    text_to_process, segmenter_obj, tokenizer_obj, reducer_obj, stop_remover_obj
):
    s_list = segmenter_obj.punkt(text_to_process)
    t_list = tokenizer_obj.pennTreeBank(s_list)
    r_list = reducer_obj.reduce(t_list)
    c_list = stop_remover_obj.fromList(r_list)

    return c_list


def build_inverted_index(docs, docIDs):
    idx = {}

    for d_idx in range(len(docs)):
        d_id = docIDs[d_idx]

        for s_idx in range(len(docs[d_idx])):
            sentence = docs[d_idx][s_idx]

            for w_idx in range(len(sentence)):
                w = sentence[w_idx]

                if w not in idx:
                    idx[w] = {}

                if d_id not in idx[w]:
                    idx[w][d_id] = 0

                idx[w][d_id] = idx[w][d_id] + 1

    return idx


def compute_idf_values(index, total_docs):
    idf_dict = {}

    for w in index:
        df = len(index[w])

        if df > 0:
            idf_dict[w] = math.log10(float(total_docs) / float(df))
        else:
            idf_dict[w] = 0.0

    return idf_dict


def compute_max_tf_per_doc(index):
    # Let us set up our required variables
    max_tf_dict = {}

    for w in index:
        for d_id in index[w]:
            tf = index[w][d_id]

            if d_id not in max_tf_dict:
                max_tf_dict[d_id] = tf
            else:
                if tf > max_tf_dict[d_id]:
                    max_tf_dict[d_id] = tf

    return max_tf_dict


def count_query_words(query):
    w_counts = {}

    for s_idx in range(len(query)):
        sentence = query[s_idx]

        for w_idx in range(len(sentence)):
            w = sentence[w_idx]

            if w not in w_counts:
                w_counts[w] = 0
            w_counts[w] = w_counts[w] + 1

    max_tf = 0
    for w in w_counts:
        if w_counts[w] > max_tf:
            max_tf = w_counts[w]

    if max_tf == 0:
        max_tf = 1

    return w_counts, max_tf


def sort_and_complete_ranking(doc_scores, all_docIDs):
    score_list = []
    for d_id in doc_scores:
        s_val = doc_scores[d_id]
        score_list.append((s_val, d_id))

    score_list.sort(reverse=True)

    ordered_ids = []
    for i in range(len(score_list)):
        ordered_ids.append(score_list[i][1])

    for d_id in all_docIDs:
        if d_id not in doc_scores:
            ordered_ids.append(d_id)

    return ordered_ids


def compute_vector_norms(concept_vectors, docIDs):
    norms_dict = {}

    for d_id in docIDs:
        sum_sq = 0.0
        c_vec = concept_vectors[d_id]

        for c_name in c_vec:
            w = c_vec[c_name]
            sum_sq = sum_sq + (w * w)

        if sum_sq > 0:
            norms_dict[d_id] = math.sqrt(sum_sq)
        else:
            norms_dict[d_id] = 0.0

    return norms_dict


def cosine_similarity_sparse(q_vec, d_vec, q_norm, d_norm):
    # Basically calculating the main metric here
    dot_p = 0.0
    for k in q_vec:
        if k in d_vec:
            dot_p = dot_p + (q_vec[k] * d_vec[k])

    if q_norm > 0 and d_norm > 0:
        cos_sim = dot_p / (q_norm * d_norm)
    else:
        cos_sim = 0.0

    return cos_sim


def parse_cranfield_data(queries_json, docs_json):
    q_ids = []
    q_texts = []
    for i in range(len(queries_json)):
        q_ids.append(queries_json[i]["query number"])
        q_texts.append(queries_json[i]["query"])

    d_ids = []
    d_texts = []
    for i in range(len(docs_json)):
        d_ids.append(docs_json[i]["id"])
        d_texts.append(docs_json[i]["body"])

    return q_ids, q_texts, d_ids, d_texts


def ranked_list_to_dict(ranked_list, query_ids):
    res = {}

    for i in range(len(query_ids)):
        res[query_ids[i]] = ranked_list[i]

    return res
