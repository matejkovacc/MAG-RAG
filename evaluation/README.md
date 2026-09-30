# FRI synthetic development pilot

`pilot.json` contains 24 assistant-authored Slovenian questions, draft reference
answers and exact supporting passages from the two-document local corpus. It
contains no historical student enquiries. All items are **development**, all
annotations are **pending human review**, and source applicability remains pending.

There are 20 answerable items, three missing-evidence/out-of-scope items and one
clarification item. Categories include direct lookup, informal wording, conditions,
exceptions, multiple passages and an amendment. Related questions share a family
ID. This is a small diagnostic pilot, not a statistically representative test set.

Do not index these questions, answers or reference passages as additional evidence.
The answering system receives only each question. Preserve the whole family in
one split when a later independently reviewed test set is designed. These exposed
pilot questions should remain development data after tuning.

Review each reference against its PDF, including exceptions and applicability.
Record changes in `review_notes`; set `review_status` to `reviewed` only after an
actual human review and supply an anonymous `reviewer_code`. References for absent
answers describe expected safe behavior, not a new university rule. Provenance
validation checks exact text and identities; it does not approve reference meaning.

The dataset includes a corpus digest and version/page/article/URL/excerpt per
reference. If the corpus changes, do not just replace the digest to make validation
pass: review and revise the affected annotations first. Separate groups in
`evidence_groups` are all required; IDs within a group are acceptable alternatives.
These are initial relevance judgments and may need correction by reviewers.

Commands and review rubric: [evaluation guide](../docs/evaluation.md).
Formal replacement of historical enquiries in the thesis still requires agreement
with the supervisor; this pilot does not establish that approval.
