"""Local public-QA adapters, deliberately separate from FRI retrieval evaluation."""

import csv
import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .controlled import write_new
from .metrics import text_metrics


class LicenseRecord(BaseModel):
    """Explicit license provenance for the data, not just its adapter code."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    dataset_name: str = Field(min_length=1)
    dataset_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    checked_on: date
    attribution: str = Field(min_length=1)
    permitted_use_confirmed: bool
    notes: str = ""


class PublicItem(BaseModel):
    """Public contextual QA or preformatted classification/choice task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    answers: list[str]
    answerable: bool
    category: str

    @model_validator(mode="after")
    def check_answers(self):
        """Unanswerable cases have no reference; other references contain text."""
        if self.answerable != bool(self.answers) or any(
            not answer.strip() for answer in self.answers
        ):
            raise ValueError("Public QA answerability and references disagree")
        return self


class PublicDataset(BaseModel):
    """Separate domain prevents treating general-language scores as office utility."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    domain: Literal["general_qa"] = "general_qa"
    format: Literal["squad2", "qaslovene"]
    license: LicenseRecord
    input_sha256: str
    items: list[PublicItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_import(self):
        """Reject unlicensed imports and duplicate IDs."""
        if not self.license.permitted_use_confirmed:
            raise ValueError("Confirm the underlying dataset license before import")
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate public QA IDs")
        return self


def import_public(
    source: Path, format: str, license_data: dict, output: Path
) -> PublicDataset:
    """Read user-downloaded SQuAD JSON or QAslovene CSV; never download or call a model."""
    license_record = LicenseRecord.model_validate(license_data)
    if not license_record.permitted_use_confirmed:
        raise ValueError("Underlying data license must be confirmed before import")
    items = []
    if format == "squad2":
        for article in json.loads(source.read_text(encoding="utf-8-sig"))["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    impossible = qa.get("is_impossible", False)
                    if type(impossible) is not bool:
                        raise ValueError("is_impossible must be a JSON boolean")
                    answers = (
                        []
                        if impossible
                        else list(
                            dict.fromkeys(answer["text"] for answer in qa["answers"])
                        )
                    )
                    if any(answer not in context for answer in answers):
                        raise ValueError(
                            "SQuAD reference text is absent from its context"
                        )
                    items.append(
                        PublicItem(
                            id=str(qa["id"]),
                            input=qa["question"] + "\n\n" + context,
                            answers=answers,
                            answerable=not impossible,
                            category="unanswerable" if impossible else "contextual_qa",
                        )
                    )
    elif format == "qaslovene":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"input", "output"} <= set(reader.fieldnames or []):
                raise ValueError("QAslovene CSV requires input/output columns")
            for index, row in enumerate(reader, 1):
                answer = row["output"].strip()
                # CSV tasks have different refusal conventions: never guess a label.
                flag = row.get("answerable", "").strip().lower()
                if flag not in {"", "true", "false"}:
                    raise ValueError("Optional answerable column must be true or false")
                if not answer and flag != "false":
                    raise ValueError(
                        "Empty output needs an explicit answerable=false annotation"
                    )
                answerable = flag != "false"
                items.append(
                    PublicItem(
                        id=row.get("id") or f"qaslovene-{index}",
                        input=row["input"],
                        answers=[answer] if answerable else [],
                        answerable=answerable,
                        category=row.get("type") or "unspecified",
                    )
                )
    else:
        raise ValueError("Unsupported public dataset format")
    dataset = PublicDataset(
        format=format,
        license=license_record,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        items=items,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "dataset.json", dataset.model_dump(mode="json"))
    with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for item in items:
            stream.write(
                json.dumps({"id": item.id, "input": item.input}, ensure_ascii=False)
                + "\n"
            )
    return dataset


def score_public(
    dataset_path: Path, predictions_path: Path, model_version: str, output: Path
) -> dict:
    """Score saved outputs only; unanswered/missing predictions are never hidden."""
    started = time.perf_counter()
    dataset = PublicDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    lookup = {item["id"]: item for item in predictions}
    if len(lookup) != len(predictions) or set(lookup) != {
        item.id for item in dataset.items
    }:
        raise ValueError("Predictions need exactly one entry per public dataset ID")
    rows = []
    for item in dataset.items:
        prediction = lookup[item.id]
        if type(prediction.get("refused")) is not bool or not isinstance(
            prediction.get("answer"), str
        ):
            raise ValueError("Each prediction needs answer:string and refused:boolean")
        if len(prediction["answer"]) > 12000 or any(
            len(answer) > 12000 for answer in item.answers
        ):
            raise ValueError("Answers exceed the bounded lexical scorer length")
        if prediction["refused"] and prediction["answer"].strip():
            raise ValueError(
                "Refused predictions must use an empty answer; keep refusal explanations outside answer"
            )
        if item.answerable:
            alternatives = [
                text_metrics(prediction["answer"], reference)
                for reference in item.answers
            ]
            metrics = {
                key: max(values[key] for values in alternatives)
                for key in alternatives[0]
            }
        else:
            metrics = {
                key: float(prediction["refused"])
                for key in ("exact_match", "token_f1", "rouge_l")
            }
        metrics["unanswerable_detection"] = (
            float(prediction["refused"]) if not item.answerable else None
        )
        metrics["false_refusal"] = (
            float(prediction["refused"]) if item.answerable else None
        )
        rows.append(
            {
                "id": item.id,
                "category": item.category,
                "answerable": item.answerable,
                "prediction": prediction,
                "metrics": metrics,
            }
        )

    def aggregate(selected):
        """Expose denominator for each general-QA metric."""
        names = sorted({key for row in selected for key in row["metrics"]})
        return {
            "n": len(selected),
            "metrics": {
                key: {"mean": mean(values) if values else None, "n": len(values)}
                for key in names
                for values in [
                    [
                        row["metrics"][key]
                        for row in selected
                        if row["metrics"][key] is not None
                    ]
                ]
            },
        }

    summary = {
        "domain": "general_qa",
        "model_version": model_version,
        "seed": None,
        "generation_configuration": "External predictions; seed and generation settings not supplied or verified by this scorer",
        "scorer": "NFKC-word-EM-multiset-F1-LCS-ROUGE-L-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "license": dataset.license.model_dump(mode="json"),
        "overall": aggregate(rows),
        "answerable": aggregate([row for row in rows if row["answerable"]]),
        "unanswerable": aggregate([row for row in rows if not row["answerable"]]),
        "by_category": {
            key: aggregate([row for row in rows if row["category"] == key])
            for key in sorted({row["category"] for row in rows})
        },
        "limits": "Local lexical diagnostics, not the official benchmark script, not FRI retrieval or practical student-office utility. No model was called by this scorer.",
    }
    summary["scoring_elapsed_seconds"] = time.perf_counter() - started
    summary["model_elapsed_seconds"] = None
    summary["worst_case_ids"] = [
        row["id"]
        for row in sorted(
            rows,
            key=lambda row: (row["metrics"]["exact_match"], row["metrics"]["token_f1"]),
        )[:10]
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", summary)
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "answerable",
                "exact_match",
                "token_f1",
                "rouge_l",
                "unanswerable_detection",
                "false_refusal",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"id": row["id"], "answerable": row["answerable"], **row["metrics"]}
            )
    (output / "report.md").write_text(
        "# General QA (separate from FRI)\n\n"
        + summary["limits"]
        + "\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return summary
