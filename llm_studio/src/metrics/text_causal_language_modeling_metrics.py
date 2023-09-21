import logging
import os
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import openai
import pandas as pd
import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sacrebleu import BLEU
from sacrebleu.metrics.base import Metric
from tenacity import retry, stop_after_attempt, wait_random_exponential
from torch import nn
from tqdm import tqdm

from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.logging_utils import TqdmToLogger

import heavyai
import csv

from itertools import permutations
import re

logger = logging.getLogger(__name__)


def sacrebleu_score(
    cfg: Any, results: Dict, val_df: pd.DataFrame, metric: Metric
) -> NDArray:
    scores = []
    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        scores.append(metric.sentence_score(predicted_text, [target_text]).score)
    return np.array(scores)


@retry(
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
)
def call_openai_api(template, model, deployment_id=None):
    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise assistant "
                "for checking the quality of the answer.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    ret = response["choices"][0]["message"]["content"]
    try:
        ret = ret.split("\n")
        score = ret[0].lower().replace("score:", "").strip().split(",")[0].split(" ")[0]
        score = float(score)
    except ValueError:
        raise ValueError(f"Could not parse score from response: {ret}")
    return score, " ".join(ret[1:]).strip()


def rate_reply(question, reference_answer, assistant_answer, model, deployment_id=None):
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/eval
    template = open("prompts/eval_template.txt", "r").read()

    template = template.format(
        question=question,
        reference_answer=reference_answer,
        assistant_answer=assistant_answer,
    )

    try:
        return call_openai_api(template, model, deployment_id)
    except Exception as e:
        logger.warning(f"Exception caught in api call: {e}")
        return 0.0, ""


def gpt_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    prompts = get_texts(val_df, cfg, separator="")

    if os.getenv("OPENAI_API_TYPE", "open_ai") == "azure":
        deployment_id = os.getenv("OPENAI_API_DEPLOYMENT_ID")
    else:
        deployment_id = None

    model = cfg.prediction.metric_gpt_model

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(rate_reply)(
            prompt,
            target_text,
            predicted_text,
            model,
            deployment_id=deployment_id,
        )
        for prompt, predicted_text, target_text in tqdm(
            zip(
                prompts,
                results["predicted_text"],
                results["target_text"],
            ),
            file=TqdmToLogger(logger, level=logging.INFO),
            desc=f"GPT eval {model}",
            total=len(prompts),
        )
    )
    scores = [x[0] for x in ret]
    explanations = [x[1] for x in ret]

    if raw_results:
        return np.array(scores), explanations
    return np.mean(scores)


#def extract_string_columns_and_literals_pairs(query):
#    pattern = r"(\w+)\s*=\s*'([^']*)'"
#    matches = re.findall(pattern, query)
#    return [{"column": match[0], "value": match[1]} for match in matches]

#def find_correct_literals(query, con):
#    string_cols_and_literals = extract_string_columns_and_literals_pairs(query)
#    for string_col_and_literal in string_cols_and_literals:
#        string_col = string_col_and_literal["column"]
#        string_literal = string_col_and_literal["value"]
#        literal_check_query = f"SELECT {string_col}, COUNT(*) AS num_matches FROM "


def extract_sql_query(text):
    if text.strip().lower().startswith("select"):
        return text
    pattern = r"SQL query:\n(.+?);"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip() + ";"
    else:
        return "No SQL query found"

def sql_rate_reply(question, db_id, query_id, gold_query, pred_query):
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/pred
    query_metadata = {
        "db_id": db_id,
        "query_id": query_id,
        "gold_query": gold_query,
        "pred_query": pred_query,
        "success": False,
        "status": "success",
        "error": None,
    }
    try:
        con = heavyai.connect(user="admin", password="HyperInteractive", host="localhost", dbname=db_id)

        extracted_gold_query = extract_sql_query(gold_query)
        gold_df = pd.read_sql(extracted_gold_query, con)
        extracted_pred_query = extract_sql_query(pred_query)
        pred_df = pd.read_sql(extracted_pred_query, con)
        num_gold_rows = len(gold_df.axes[0])
        num_pred_rows = len(pred_df.axes[0])
        num_gold_cols = len(gold_df.axes[1])
        num_pred_cols = len(pred_df.axes[1])

        if num_gold_rows != num_pred_rows:
            print("ROW COUNT MISMATCH")
            print(extracted_gold_query)
            print(extracted_pred_query)
            query_metadata["status"] = "row_count_mismatch"
            return 0.0, query_metadata 
        if num_gold_cols != num_pred_cols:
            print("COL COUNT MISMATCH")
            print(extracted_gold_query)
            print(extracted_pred_query)
            query_metadata["status"] = "col_count_mismatch"
            return 0.0, query_metadata 
        gold_query_has_order_by = gold_query.lower().find("order by") >= 0  
        dfs_are_equal = None
        if gold_query_has_order_by:
            dfs_are_equal = np.array_equal(gold_df.values, pred_df.values)
        else:
            gold_df_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
            pred_df_sorted = pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True)
            dfs_are_equal = np.array_equal(gold_df_sorted.values, pred_df_sorted.values)
            #dfs_are_equal = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True).equals(pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True))
        if dfs_are_equal:
            query_metadata["success"] = True
            return 1.0, query_metadata 
        else:
            for cols in permutations(pred_df.columns):
                pred_df_perm = pred_df[list(cols)]
                if gold_query_has_order_by:
                    dfs_are_equal = np.array_equal(gold_df.values, pred_df_perm.values)
                else:
                    gold_df_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
                    pred_df_perm_sorted = pred_df_perm.sort_values(by=list(pred_df_perm.columns)).reset_index(drop=True)
                    dfs_are_equal = np.array_equal(gold_df_sorted.values, pred_df_perm_sorted.values)
                if dfs_are_equal:
                    print("COLUMN ORDER DIFF")
                    query_metadata["success"] = True
                    query_metadata["status"] = "column_order_difference"
                    return 1.0, query_metadata 
            print("VALUES MISMATCH")
            query_metadata["status"] = "values_mismatch"
            print(extracted_gold_query)
            print(extracted_pred_query)
            return 0.0, query_metadata 
    except Exception as e:
        print("QUERY FAIL")
        query_metadata["status"] = "execution_error"
        query_metadata["error"] = e 
        return 0.0, query_metadata 

def write_sql_metadata_to_csv(metadata, output_file):
    fieldnames = [
        "query_id",
        "db_id",
        "gold_query",
        "pred_query",
        "success"
        "status",
        "error"
    ]
    metadata_to_write = [
        (
            obj["query_id"],
            obj["db_id"],
            obj["gold_query"],
            obj["pred_query"],
            obj["success"],
            obj["status"],
            obj["error"],
        )
        for obj in metadata
    ]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(metadata_to_write)


def sql_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:

    prompts = get_texts(val_df, cfg, separator="")
    db_ids = None
    db_ids = val_df["db_id"].astype(str)
    db_ids = db_ids.values
    query_ids = None
    if "id" in val_df:
        query_ids = val_df["id"].astype(str)
    else:
        query_ids = val_df["query_id"].astype(str)
    query_ids = query_ids.values

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(sql_rate_reply)(
            prompt,
            db_id,
            query_id,
            target_text,
            predicted_text,
        )
        for prompt, db_id, query_id, predicted_text, target_text in tqdm(
            zip(
                prompts,
                db_ids,
                query_ids,
                results["predicted_text"],
                results["target_text"],
            ),
            file=TqdmToLogger(logger, level=logging.INFO),
            desc=f"SQL eval",
            total=len(prompts),
        )
    )

    scores = [x[0] for x in ret]
    explanations = [x[1]["status"] for x in ret]
    query_metadata = [x[1] for x in ret]
    write_sql_metadata_to_csv(query_metadata, "/home/ubuntu/sql_eval.log")


    if raw_results:
        return np.array(scores), explanations
    return np.array(scores)
    #return np.mean(scores)


class Perplexity(nn.Module):
    def __init__(self, cfg: Any, reduce: bool = True):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


def perplexity(cfg: Any, results: Dict, val_df: pd.DataFrame):
    return results["perplexity"].detach().float().cpu().numpy()


class Metrics:
    """
    Metrics factory. Returns:
        - metric value
        - should it be maximized or minimized
        - Reduce function

    Maximized or minimized is needed for early stopping (saving best checkpoint)
    Reduce function to generate a single metric value, usually "mean" or "none"
    """

    _metrics = {
        "Perplexity": (perplexity, "min", "mean"),
        "BLEU": (
            partial(sacrebleu_score, metric=BLEU(effective_order=True)),
            "max",
            "mean",
        ),
        "GPT": (gpt_score, "max", "mean"),
        "SQL": (sql_score, "max", "mean")
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Metrics.

        Args:
            name: metrics name
        Returns:
            A class to build the Metrics
        """
        return cls._metrics.get(name, "GPT")

    @classmethod
    def suitable_metrics(cls, cfg: Any, results: Dict, val_df: pd.DataFrame) -> Dict:
        """Access to All Suitable Metrics. For some problem types (e.g. classification)
        there might be metrics (e.g. Micro Averaged F1) that are only suitable in
        specific cases (multiclass not binary). There might also be additional
        metrics returned, which are not possible to select as validation metrics,
        e.g. threshold dependant metrics

        Returns:
            A dictionary of all suitable metrics for current problem setup
        """
        return cls._metrics

    @classmethod
    def all_metrics(cls) -> Dict:
        """Access to All Metrics. There might also be additional
        metrics returned, which are not possible to select as validation metrics,
        e.g. threshold dependant metrics

        Returns:
            A dictionary of all metrics (including not suitable metrics).
        """
        return cls._metrics
