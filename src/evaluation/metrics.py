"""Deterministic lexical QA metrics with explicit denominators and no model calls."""

import re
import unicodedata
from collections import Counter


def tokens(text: str) -> list[str]:
    """Casefold Unicode words; retain Slovenian diacritics and do not stem words."""
    return re.findall(
        r"\w+", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE
    )


def normalized(text: str) -> str:
    """Normalize punctuation and whitespace for exact-match and duplicate checks."""
    return " ".join(tokens(text))


def text_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Return normalized EM, multiset token F1 and token LCS ROUGE-L F1."""
    predicted, expected = tokens(prediction), tokens(reference)
    if not expected:
        raise ValueError("Reference answers must not be empty")
    common = sum((Counter(predicted) & Counter(expected)).values())
    f1 = 2 * common / (len(predicted) + len(expected)) if predicted else 0.0
    # Linear memory, quadratic time; callers bound output/reference lengths.
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, other in enumerate(expected, 1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    rouge = 2 * previous[-1] / (len(predicted) + len(expected)) if predicted else 0.0
    return {
        "exact_match": float(predicted == expected),
        "token_f1": f1,
        "rouge_l": rouge,
    }


def retrieval_scores(
    source_ids: list[str], ranked_ids: list[str], groups: list[list[str]] | None = None
) -> dict:
    """Distinguish any-hit rate from fraction of all required chunks retrieved."""
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Duplicate retrieved source IDs")
    required = set(source_ids)
    groups = groups if groups is not None else [[key] for key in source_ids]
    scores = {}
    for k in (1, 3, 5):
        found = required.intersection(ranked_ids[:k])
        scores[f"hit_rate_at_{k}"] = float(bool(found)) if required else None
        scores[f"recall_at_{k}"] = (
            sum(bool(found.intersection(group)) for group in groups) / len(groups)
            if groups
            else None
        )
    rank = next(
        (
            index
            for index, identifier in enumerate(ranked_ids, 1)
            if identifier in required
        ),
        None,
    )
    scores["reciprocal_rank"] = (1 / rank if rank else 0.0) if required else None
    scores["any_correct_source"] = (
        float(bool(required.intersection(ranked_ids))) if required else None
    )
    return scores


def answer_scores(item, answer) -> dict:
    """Separate word overlap, refusal behavior and source-ID correctness proxies."""
    response = "\n".join(part.text for part in answer.parts) if answer else ""
    scores = (
        text_metrics(response, item.expected_answer)
        if item.answerable
        else {key: None for key in ("exact_match", "token_f1", "rouge_l")}
    )
    actual = answer.status if answer else None
    scores["status_match"] = float(actual == item.expected_status)
    scores["unanswerable_detection"] = (
        float(actual == "no_evidence")
        if not item.answerable and item.expected_status == "no_evidence"
        else None
    )
    scores["false_refusal"] = (
        float(actual == "no_evidence") if item.answerable else None
    )
    cited = (
        {identifier for part in answer.parts for identifier in part.citation_ids}
        if answer
        else set()
    )
    correct = cited.intersection(item.source_ids)
    scores["answer_with_correct_source"] = (
        float(bool(correct)) if item.answerable else None
    )
    scores["cited_source_precision"] = (
        len(correct) / len(cited) if cited and item.answerable else None
    )
    scores["required_source_citation_recall"] = (
        sum(bool(correct.intersection(group)) for group in item.required_source_groups)
        / len(item.required_source_groups)
        if item.answerable
        else None
    )
    return scores
