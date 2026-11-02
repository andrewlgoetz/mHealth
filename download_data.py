"""Download the two original releases and keep their documentation with the data."""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep the publisher URLs here so the original releases are easy to identify.
SOURCES = {
    "uci_har": {
        "url": (
            "https://archive.ics.uci.edu/static/public/240/"
            "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
        ),
        "archive": "uci_har.zip",
        "page": (
            "https://archive.ics.uci.edu/dataset/240/"
            "human+activity+recognition+using+smartphones"
        ),
    },
    "wisdm": {
        "url": (
            "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/"
            "WISDM_ar_latest.tar.gz"
        ),
        "archive": "WISDM_ar_latest.tar.gz",
        "page": "https://www.cis.fordham.edu/wisdm/dataset.php",
    },
}


def extract_archive(archive, destination):
    """Extract ordinary files only, keeping every path inside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()

                # Archive filenames must not write outside the data directory.
                if not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive path: {member.filename}")

                if not member.is_dir():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        bundle.open(member) as source_file,
                        target.open("wb") as output_file,
                    ):
                        shutil.copyfileobj(source_file, output_file)

    else:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                target = (destination / member.name).resolve()

                if not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive path: {member.name}")

                # Copy regular files only; do not create links from the archive.
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        bundle.extractfile(member) as source_file,
                        target.open("wb") as output_file,
                    ):
                        shutil.copyfileobj(source_file, output_file)


def download_all(data_dir=ROOT / "data", report_dir=ROOT / "reports"):
    """Reuse existing downloads, verify their hashes, and extract each release."""
    data_dir = Path(data_dir)
    report_dir = Path(report_dir)
    archives = data_dir / "downloads"
    archives.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # The manifest records which exact files were used for the experiments.
    manifest_path = report_dir / "data_sources.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
    else:
        previous = {}

    manifest = previous.copy()

    for name, source in SOURCES.items():
        archive = archives / source["archive"]

        if not archive.exists():
            print(f"Downloading {name} from {source['url']}", flush=True)
            request = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "mHealth-research/1.0"},
            )

            # A partial download must not be mistaken for a complete archive.
            temporary = archive.with_suffix(archive.suffix + ".part")
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output_file,
            ):
                shutil.copyfileobj(response, output_file)

            temporary.replace(archive)

        with archive.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()

        # Stop if a previously recorded archive has changed between runs.
        if name in previous and previous[name]["sha256"] != checksum:
            raise ValueError(
                f"{archive} differs from the recorded SHA-256. "
                "Check the release before replacing the manifest."
            )

        destination = data_dir / "raw" / name
        marker = destination / ".extracted"

        if not marker.exists():
            print(f"Extracting {name}", flush=True)
            extract_archive(archive, destination)

            # UCI's current download wraps the original ZIP inside another ZIP.
            if (
                name == "uci_har"
                and not list(destination.rglob("total_acc_x_train.txt"))
            ):
                nested_archives = list(destination.rglob("UCI HAR Dataset.zip"))
                if len(nested_archives) != 1:
                    raise ValueError(
                        "Cannot find the UCI HAR inertial signals or original archive."
                    )

                extract_archive(nested_archives[0], destination)

            if name == "uci_har":
                expected_file = "total_acc_x_train.txt"
            else:
                expected_file = "WISDM_ar_v1.1_raw.txt"

            if len(list(destination.rglob(expected_file))) != 1:
                raise ValueError(
                    f"Expected one {expected_file}; check the downloaded release."
                )

            # Write the marker only after extraction and file checks succeed.
            marker.write_text(checksum + "\n", encoding="utf-8")

        if marker.read_text().strip() != checksum:
            raise ValueError(f"{destination} was extracted from a different archive.")

        manifest[name] = {
            **source,
            "sha256": checksum,
            "bytes": archive.stat().st_size,
            "downloaded_utc": previous.get(name, {}).get(
                "downloaded_utc",
                datetime.now(timezone.utc).isoformat(),
            ),
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: {archive.stat().st_size:,} bytes, SHA-256 {checksum}",
            flush=True,
        )

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    download_all(args.data_dir)
