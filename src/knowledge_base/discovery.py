"""Section-aware discovery of regulations from the official FRI landing page."""

import hashlib
import json
import re
import unicodedata
from datetime import date
from urllib.parse import (
    parse_qs,
    quote,
    unquote,
    urldefrag,
    urljoin,
    urlsplit,
    urlunsplit,
)

from src.knowledge_base.models import RegulationMetadata
from src.knowledge_base.website import (
    ContentParser,
    Element,
    descendants,
    ignored,
    normalize,
    plain_text,
)

LANDING_PAGE = "https://fri.uni-lj.si/sl/pravilniki-vloge-ceniki"
OFFICIAL_HOSTS = frozenset(
    {
        "fri.uni-lj.si",
        "www.fri.uni-lj.si",
        "uni-lj.si",
        "www.uni-lj.si",
        "pisrs.si",
        "www.pisrs.si",
    }
)
NORMATIVE = re.compile(
    r"pravilnik|statut|študijsk\w* red|navodil|interpretacij|usmerit", re.I
)
FORM = re.compile(
    r"^(?:prošnj|vlog|prijav|odjav|soglasj|izjav|zapisnik|ocena dispozicije|seminar\s)|predlog[aeo]|obrazec",
    re.I,
)
AMENDMENT = re.compile(r"sprememb|dopolnit", re.I)


def normalized_title(text: str) -> str:
    """Normalize layout/case, preserving meaningful version and date labels."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def canonical_url(url: str) -> str:
    """Encode spaces/Slovenian characters once and remove non-resource fragments."""
    parts = urlsplit(urldefrag(url)[0])
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            quote(unquote(parts.path), safe="/!$&'()*+,;=:@-._~"),
            quote(unquote(parts.query), safe="=&?/:;+,%@-._~"),
            "",
        )
    )


def source_key(title: str, section: str) -> str:
    """Keep logical identity stable across URL changes, independent of link order."""
    identity = normalized_title(section) + "\n" + normalized_title(title)
    return "fri-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def document_type(url: str) -> str:
    """Infer candidate type from its path; the downloader also validates MIME/magic."""
    suffix = urlsplit(url).path.rsplit(".", 1)[-1].lower()
    return suffix if suffix in {"pdf", "doc", "docx"} else "html"


def labelled_date(text: str, label: str) -> date | None:
    """Read only explicitly labelled dates, never dates guessed from filenames."""
    match = re.search(
        label + r"\s*:?\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})", text, re.I
    )
    if not match:
        return None
    try:
        return date(int(match[3]), int(match[2]), int(match[1]))
    except ValueError:
        return None


def own_text(node: Element) -> str:
    """Read a list item's label without inheriting nested amendments or forms."""
    return normalize(
        plain_text(
            Element(
                node.tag,
                node.attrs,
                [
                    child
                    for child in node.children
                    if not isinstance(child, Element) or child.tag not in {"ul", "ol"}
                ],
            )
        )
    )


def subcategory(title: str, section: str) -> str:
    """Assign descriptive families without deciding legal applicability."""
    value = normalized_title(title)
    for marker, name in (
        ("študijsk", "study_rules"),
        ("statut", "statute"),
        ("višji letnik", "student_status"),
        ("disciplinsk", "disciplinary"),
        ("tutor", "tutoring"),
        ("izmenjav", "international_exchange"),
        ("prešern", "preseren_awards"),
    ):
        if marker in value and (marker != "študijsk" or "red" in value):
            return name
    return section


def discover_regulations(content: bytes, url: str = LANDING_PAGE) -> list[dict]:
    """Inventory selected-body links with inclusion/exclusion reasons and metadata.

    Supports semantic headings and FRI's styled span/strong headings. Nested list
    items retain their own labels and immediate parent regulation. Unrecognized
    normative links remain visible as exclusions rather than disappearing.
    """
    parser = ContentParser()
    parser.feed(content.decode("utf-8-sig"))
    roots = [
        n for n in descendants(parser.root) if n.attrs.get("id") == "katedre-container"
    ]
    if len(roots) != 1:
        raise ValueError("Discovery expected one FRI content container")
    section = "outside"
    records = []
    seen = set()

    def visit(node: Element, context: str = "", parent: str | None = None) -> None:
        """Walk the visible content in order, tracking sections and nested lists."""
        nonlocal section
        if ignored(node):
            return
        text = normalized_title(plain_text(node))
        heading = node.tag in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "strong",
        } or "naslov-test" in node.attrs.get("class", "")
        if heading and not any(
            n.tag == "a" and n.attrs.get("href") for n in descendants(node)
        ):
            if text == "pravilniki":
                section = "general"
            elif "zaključna dela" in text or "zaključnih del" in text:
                section = "thesis"
            elif "prva stopnja" in text:
                section = "bachelor_thesis"
            elif "druga stopnja" in text:
                section = "master_thesis"
            elif "doktorski študij" in text or "tretja stopnja" in text:
                section = "doctoral"
            elif "cenik" in text or "prošnje" in text or "vloge" in text:
                section = "excluded"
        if node.tag == "li":
            context = own_text(node)
        child_parent = parent
        if node.tag == "a" and node.attrs.get("href"):
            target = canonical_url(urljoin(url, node.attrs["href"]))
            title = normalize(plain_text(node))
            label = (
                context
                if normalized_title(title) in {"pdf", "doc", "docx", "tex"}
                else title
            )
            # File-type labels are presentation, not the meaning of a link.
            label = re.sub(r"\s*\[\s*PDF.*$", "", label, flags=re.I).strip()
            kind = document_type(target)
            reason = "normative document in regulation section"
            include = True
            if section in {"outside", "excluded"}:
                include, reason = (
                    False,
                    "outside regulation sections / forms or price lists",
                )
            elif FORM.search(label) or re.search(
                r"(?:^|[/_])obrazec[_ .]", unquote(urlsplit(target).path), re.I
            ):
                include, reason = False, "form or template (label and URL context)"
            elif not NORMATIVE.search(normalized_title(label)):
                include, reason = False, "not a normative or instructional document"
            elif (
                urlsplit(target).hostname not in OFFICIAL_HOSTS
                or urlsplit(target).scheme != "https"
            ):
                include, reason = (
                    False,
                    "unsupported official destination; review required",
                )
            elif target in seen:
                include, reason = False, "duplicate URL"
            key = source_key(label, section)
            metadata = RegulationMetadata(
                landing_page=url,
                linked_url=target,
                document_type=kind,
                category=(
                    "instruction"
                    if re.match(r"navodil|usmerit", label, re.I)
                    else "regulation"
                ),
                subcategory=subcategory(label, section),
                publication_date=labelled_date(context, r"objavljen[oa]"),
                effective_date=labelled_date(context, r"velja(?:jo)? od"),
                version_label=(
                    label
                    if re.search(r"prečiščeno|sprememb|dopolnit|\d{4}", label, re.I)
                    else None
                ),
                is_amendment=bool(AMENDMENT.search(label)),
                parent_regulation=parent if AMENDMENT.search(label) else None,
                discovery_path=[url],
            )
            records.append(
                dict(
                    key=key,
                    title=label,
                    url=target,
                    format=kind,
                    section=section,
                    included=include,
                    reason=reason,
                    regulation=metadata.model_dump(mode="json"),
                )
            )
            if include:
                seen.add(target)
        # A nested list's parent is the direct preceding link, never a form sibling.
        for child in node.children:
            if isinstance(child, Element):
                visit(child, context, child_parent)
                if child.tag == "a" and records and records[-1]["included"]:
                    child_parent = records[-1]["key"]

    visit(roots[0])
    included = [r for r in records if r["included"]]
    required = {"general", "bachelor_thesis", "master_thesis", "doctoral"}
    if not required <= {r["section"] for r in included}:
        raise ValueError(
            "Discovery missing a required regulation section; refusing a reduced corpus"
        )
    # Same title for distinct documents must never silently overwrite a sibling.
    for record in included:
        if sum(other["key"] == record["key"] for other in included) > 1:
            raise ValueError(
                "Ambiguous duplicate regulation title; review source identities"
            )
    return records


def official_document_links(content: bytes, url: str, family: str) -> list[dict]:
    """Resolve a UL regulation portal to matching current PDFs, excluding archives.

    Retain both consolidated texts and amendments/base versions listed in the
    current section. Never infer that a base version has been repealed merely
    because a consolidated document is also present.
    """
    parser = ContentParser()
    parser.feed(content.decode("utf-8-sig"))
    roots = [n for n in descendants(parser.root) if n.tag == "main"]
    if len(roots) != 1:
        raise ValueError("Official portal has no unique main content")
    matches = []
    archive = False
    for node in descendants(roots[0]):
        title = " ".join(normalize(plain_text(node)).split())
        folded = normalized_title(title)
        if node.tag in {"h1", "h2", "h3", "h4"}:
            if "arhiv" in folded:
                archive = True
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        target = canonical_url(urljoin(url, node.attrs["href"]))
        if archive or "arhiv" in unquote(target).casefold():
            continue
        if (
            document_type(target) != "pdf"
            or urlsplit(target).hostname not in OFFICIAL_HOSTS
        ):
            continue
        title = re.sub(r"\s+pdf\s+[\d.,]+\s*[KM]B$", "", title, flags=re.I)
        if (
            family == "doctoral" and re.search(r"pravilnik\w* o doktorskem", folded)
        ) or (family == "disciplinary" and "disciplinsk" in folded):
            matches.append(dict(title=title, url=target))
    return list({r["url"]: r for r in matches}.values())


def navigation_link(content: bytes, url: str, label: str) -> str:
    """Follow one exact official directory label when an old UL route redirects."""
    parser = ContentParser()
    parser.feed(content.decode("utf-8-sig"))
    targets = {
        canonical_url(urljoin(url, node.attrs["href"]))
        for node in descendants(parser.root)
        if node.tag == "a"
        and node.attrs.get("href")
        and normalized_title(plain_text(node)) == normalized_title(label)
    }
    if len(targets) != 1:
        raise ValueError(f"Official redirect recovery needs one '{label}' link")
    return targets.pop()


def pisrs_document(url: str, capture) -> dict:
    """Resolve a PISRS record through its public API to the latest NPB PDF export.

    API routes are the same routes used by the official site's client. Regulation,
    version and file IDs come from the landing link/returned metadata, never a
    hardcoded statute or PDF URL. Older NPB identities remain in the raw capture.
    """
    parts = urlsplit(url)
    ids = parse_qs(parts.query).get("id", [])
    if (
        parts.path != "/pregledPredpisa"
        or len(ids) != 1
        or not re.fullmatch(r"[A-Za-z0-9]+", ids[0])
    ):
        raise ValueError("Unsupported PISRS regulation link")
    origin = f"https://{parts.hostname}"
    config_url = origin + "/assets/config.json"
    config = json.loads(capture(config_url).content)
    api = config.get("BACKEND_ENDPOINT", "").rstrip("/")
    if urlsplit(api).hostname != parts.hostname or urlsplit(api).scheme != "https":
        raise ValueError("PISRS public API must stay on the linked HTTPS host")
    record_url = api + "/rezultat/zbirka/id/" + ids[0]
    data = json.loads(capture(record_url).content).get("data", {})
    evidence = data.get("evidencniPodatki", {})
    if evidence.get("zunanjiID") != ids[0] or evidence.get("veljaDo"):
        raise ValueError("PISRS identity mismatch or regulation marked ended")
    versions = data.get("besedilo", {}).get("npbVerzije", [])
    numbered = [
        (int(v["naziv"].split()[1]), v)
        for v in versions
        if re.fullmatch(r"NPB \d+", v.get("naziv", ""))
    ]
    if not numbered:
        raise ValueError("PISRS has no consolidated version; review required")
    latest = max(numbered, key=lambda pair: pair[0])[1]
    files = [
        f
        for group in data.get("datoteke", [])
        if group.get("npbVerzija", {}).get("id") == latest["id"]
        for f in group.get("datoteke", [])
        if f.get("tip") == "PDF_DOCUMENT"
    ]
    if len(files) != 1 or not isinstance(files[0].get("id"), int):
        raise ValueError("PISRS latest version has no unique PDF export")
    return {
        "url": api + "/datoteke/integracije/" + str(files[0]["id"]),
        "version_label": latest["naziv"],
        "discovery_path": [url, config_url, record_url],
        "available_versions": versions,
    }
