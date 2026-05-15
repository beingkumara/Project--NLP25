from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

# Importing the utility functions we need for converting formats
from util import get_relevant_docs, ranked_list_to_dict

from sys import version_info
import argparse
import json
import matplotlib

matplotlib.use("Agg")  # Need this so matplotlib does not try to open a display window
import matplotlib.pyplot as plt
import os

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, args):
        self.args = args

        # Create output folder if it does not exist yet
        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder)

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()

    def segmentSentences(self, text):
        # we iterate through the logic here
        """
        Basically this function segments the input raw text into sentences.
        It takes the text as a string and returns a list of sentence strings.
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Here we tokenize the input sentences.
        Takes a list of strings and returns a list of token lists.
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        # we iterate through the logic here
        """
        From what we studied, we need to reduce the words to their root forms here.
        Returns the reduced tokens.
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Let us remove the irrelevant common stop words from the text.
        Returns the tokens without stopwords.
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Basically the complete preprocessing pipeline for our queries.
        Returns the final preprocessed queries.
        """
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)

        out_file = open(
            os.path.join(self.args.out_folder, "segmented_queries.txt"), "w"
        )
        json.dump(segmentedQueries, out_file)
        out_file.close()

        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)

        out_file = open(
            os.path.join(self.args.out_folder, "tokenized_queries.txt"), "w"
        )
        json.dump(tokenizedQueries, out_file)
        out_file.close()

        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)

        out_file = open(os.path.join(self.args.out_folder, "reduced_queries.txt"), "w")
        json.dump(reducedQueries, out_file)
        out_file.close()

        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)

        out_file = open(
            os.path.join(self.args.out_folder, "stopword_removed_queries.txt"), "w"
        )
        json.dump(stopwordRemovedQueries, out_file)
        out_file.close()

        return stopwordRemovedQueries

    def preprocessDocs(self, docs):
        """
        Here is the complete preprocessing pipeline for our documents.
        Returns the final preprocessed documents.
        """
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)

        out_file = open(os.path.join(self.args.out_folder, "segmented_docs.txt"), "w")
        json.dump(segmentedDocs, out_file)
        out_file.close()

        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)

        out_file = open(os.path.join(self.args.out_folder, "tokenized_docs.txt"), "w")
        json.dump(tokenizedDocs, out_file)
        out_file.close()

        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)

        out_file = open(os.path.join(self.args.out_folder, "reduced_docs.txt"), "w")
        json.dump(reducedDocs, out_file)
        out_file.close()

        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)

        out_file = open(
            os.path.join(self.args.out_folder, "stopword_removed_docs.txt"), "w"
        )
        json.dump(stopwordRemovedDocs, out_file)
        out_file.close()

        return stopwordRemovedDocs

    def evaluateDataset(self):
        """
        Finally running the full IR system evaluation on the Cranfield dataset.
        """

        # Loading query data from the dataset folder
        queries_file = open(os.path.join(self.args.dataset, "cran_queries.json"), "r")
        queries_json = json.load(queries_file)
        queries_file.close()

        # Extracting query ids and the actual query text from the loaded data
        query_ids = []
        queries = []
        for item in queries_json:
            query_ids.append(item["query number"])
            queries.append(item["query"])

        processedQueries = self.preprocessQueries(queries)

        # Loading documents from the dataset folder
        docs_file = open(os.path.join(self.args.dataset, "cran_docs.json"), "r")
        docs_json = json.load(docs_file)
        docs_file.close()

        # Extracting document ids and body text
        doc_ids = []
        docs = []
        for item in docs_json:
            doc_ids.append(item["id"])
            docs.append(item["body"])

        processedDocs = self.preprocessDocs(docs)

        # Building the index and then ranking the processed queries
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        doc_IDs_ordered_list = self.informationRetriever.rank(processedQueries)

        # Converting the ranked list into a dict so evaluation functions can look up by query_id.
        # The evaluation module expects {query_id: [ranked_doc_ids]}, not a plain list.
        doc_IDs_ordered = ranked_list_to_dict(doc_IDs_ordered_list, query_ids)

        # Loading the qrels relevance file
        qrels_file = open(os.path.join(self.args.dataset, "cran_qrels.json"), "r")
        qrels_raw = json.load(qrels_file)
        qrels_file.close()

        # Converting qrels from the raw JSON list into a dict of {query_id: [relevant_doc_ids]}.
        # The evaluation module expects this dict format, not the raw list.
        # A document is considered relevant if its position in qrels is 1, 2, 3, or 4.
        qrels = get_relevant_docs(qrels_raw)

        # Some queries may not have any relevant documents in the qrels file.
        # We initialize those with an empty list to avoid KeyError during evaluation.
        for q_id in query_ids:
            if q_id not in qrels:
                qrels[q_id] = []

        precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []

        for k in range(1, 11):

            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

            print(
                "Precision, Recall, F-score @ "
                + str(k)
                + ": "
                + str(precision)
                + ", "
                + str(recall)
                + ", "
                + str(fscore)
            )

            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k
            )
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            MRR = self.evaluator.meanReciprocalRank(
                doc_IDs_ordered, query_ids, qrels, k
            )

            MAPs.append(MAP)
            nDCGs.append(nDCG)
            MRRs.append(MRR)

            print(
                "MAP, nDCG, MRR @ "
                + str(k)
                + ": "
                + str(MAP)
                + ", "
                + str(nDCG)
                + ", "
                + str(MRR)
            )

        # Now plotting all the metrics together on one graph
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.plot(range(1, 11), MRRs, label="MRR")

        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(os.path.join(self.args.out_folder, "eval_plot.png"))
        plt.close()

    def handleCustomQuery(self):
        """
        This is to allow the user to enter a custom query via the terminal logic.
        """

        print("Enter query below")
        query = input()

        processedQuery = self.preprocessQueries([query])[0]

        # Loading documents so we can build the index for ranking
        docs_file = open(os.path.join(self.args.dataset, "cran_docs.json"), "r")
        docs_json = json.load(docs_file)
        docs_file.close()

        # Extracting document ids and body text
        doc_ids = []
        docs = []
        for item in docs_json:
            doc_ids.append(item["id"])
            docs.append(item["body"])

        processedDocs = self.preprocessDocs(docs)

        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # rank() returns a list of lists, so we take [0] to get results for the single query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Build a lookup dict so we can print doc details by id
        doc_lookup = {}
        for item in docs_json:
            doc_lookup[item["id"]] = item

        print("\nTop 5 Results:\n" + "=" * 60)
        for rank, id_ in enumerate(doc_IDs_ordered[:5], start=1):
            doc = doc_lookup.get(id_, {})
            title = doc.get("title", "N/A").strip()
            author = doc.get("author", "N/A").strip()
            body = doc.get("body", "").strip()
            snippet = body[:250] + "..." if len(body) > 250 else body
            print(f"Rank {rank} | Doc ID: {id_}")
            print(f"Title  : {title}")
            print(f"Author : {author}")
            print(f"Snippet: {snippet}")
            print("-" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="main.py")

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("-dataset", default=os.path.join(_script_dir, "cranfield/"))
    parser.add_argument("-out_folder", default=os.path.join(_script_dir, "output/"))
    parser.add_argument("-segmenter", default="punkt")
    parser.add_argument("-tokenizer", default="ptb")
    parser.add_argument("-custom", action="store_true")

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
