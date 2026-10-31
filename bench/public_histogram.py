"""Derive a reproducible invoice-length workload from licensed UCI data.

Only lengths and their histogram are published. Invoice identifiers, customer
identifiers and product text are not copied to the output. A segment contains
all source rows sharing an InvoiceNo, including cancellations and duplicates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from bench.distributions import describe

SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
SOURCE_SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
DATASET_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def extract_lengths(archive: Path) -> np.ndarray:
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("UCI archive checksum differs from the recorded source")
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        xlsx = outer.read("Online Retail.xlsx")
    with zipfile.ZipFile(io.BytesIO(xlsx)) as book:
        shared = ET.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(item.itertext()) for item in shared]
        invoices = Counter()
        header = None
        with book.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != NS + "row":
                    continue
                first = next((c for c in row if c.attrib.get("r", "").rstrip("0123456789") == "A"), None)
                if first is None:
                    raise ValueError("source row lacks InvoiceNo")
                value = first.find(NS + "v")
                if value is None or value.text is None:
                    raise ValueError("source row has empty InvoiceNo")
                invoice = strings[int(value.text)] if first.attrib.get("t") == "s" else value.text
                if header is None:
                    header = invoice
                    if header != "InvoiceNo":
                        raise ValueError("unexpected first column/header")
                else:
                    invoices[invoice] += 1
                row.clear()
    lengths = np.asarray(list(invoices.values()), dtype=np.int64)
    if int(lengths.sum()) != 541909 or len(lengths) != 25900:
        raise ValueError("source row/invoice totals differ from the pinned extraction")
    return lengths


def sample_histogram(path: Path, segments: int, *, seed: int, total_elements: int | None) -> np.ndarray:
    metadata = json.loads(path.read_text())
    pairs = metadata.get("length_counts")
    if (not isinstance(pairs, list) or not pairs
            or any(not isinstance(p, list) or len(p) != 2
                   or any(type(x) is not int or x <= 0 or x > 2**63-1 for x in p) for p in pairs)):
        raise ValueError("histogram must contain positive integer [length, count] pairs")
    if type(segments) is not int or segments < 0 or type(seed) is not int or seed < 0:
        raise ValueError("segments and seed must be nonnegative integers")
    if total_elements is not None and (type(total_elements) is not int or not 0 <= total_elements <= 2**63-1):
        raise ValueError("total_elements must be a nonnegative signed-i64 integer")
    if segments == 0:
        if total_elements not in (0, None):
            raise ValueError("nonzero elements cannot belong to zero segments")
        return np.zeros(0, dtype=np.int64)
    frequency = np.asarray([p[1] for p in pairs], dtype=np.float64)
    lengths = np.random.default_rng(seed).choice(np.asarray([p[0] for p in pairs], dtype=np.int64),
                                               size=segments, p=frequency / frequency.sum())
    denominator = sum(map(int, lengths))
    if total_elements is None:
        if denominator > 2**63-1:
            raise ValueError("sampled element count exceeds signed-i64")
        return lengths
    numerators = [int(length) * total_elements for length in lengths]
    scaled = [x // denominator for x in numerators]
    remainder = total_elements - sum(scaled)
    order = sorted(range(segments), key=lambda i: (-(numerators[i] % denominator), i))
    for i in order[:remainder]:
        scaled[i] += 1
    return np.asarray(scaled, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = extract_lengths(args.archive)
    metadata = describe(lengths) | {
        "schema": 1, "dataset": "UCI Online Retail", "dataset_url": DATASET_URL,
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "attribution": "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33",
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_observed_date": "2026-10-13", "license_evidence_url": DATASET_URL,
        "transformation": "Count source rows per InvoiceNo; preserve cancellations and duplicate rows; order by first occurrence; export only lengths",
        "length_counts": [[int(value), int(count)] for value, count in sorted(Counter(lengths).items())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), lengths=lengths)
    print(json.dumps({k: metadata[k] for k in ("segments", "elements", "min_length", "max_length", "lengths_sha256")}, indent=2))


if __name__ == "__main__":
    main()
