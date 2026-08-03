"""Download the Piper binary and a voice into ``models/``.

    uv run python scripts/fetch_models.py

The repository this replaces vendored 289 MB of Windows binaries and ONNX voices
directly in git, which is both unpushable to GitHub and unusable on Linux. They
are fetched on demand instead, into a gitignored directory.

If Piper is already installed elsewhere, skip this and point the config at it:

    VA_PIPER_BIN=/path/to/piper  VA_PIPER_VOICE=/path/to/voice.onnx
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PIPER_VERSION = "2023.11.14-2"
PIPER_RELEASES = f"https://github.com/rhasspy/piper/releases/download/{PIPER_VERSION}"

#: Release asset per platform. Piper ships prebuilt binaries; building from
#: source pulls in espeak-ng and onnxruntime and is not worth it here.
PIPER_ASSETS = {
    ("Windows", "AMD64"): "piper_windows_amd64.zip",
    ("Linux", "x86_64"): "piper_linux_x86_64.tar.gz",
    ("Linux", "aarch64"): "piper_linux_aarch64.tar.gz",
    ("Darwin", "x86_64"): "piper_macos_x64.tar.gz",
    ("Darwin", "arm64"): "piper_macos_aarch64.tar.gz",
}

VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

#: Medium quality is the right default: the high-quality models are noticeably
#: slower to synthesise and the difference is inaudible at telephone bandwidth.
VOICES = {
    # English, US
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-ryan-medium": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    # English, GB. The demo domain is a UK builders' merchant, so a British
    # voice is not decoration — an American voice saying "Severn Valley" and
    # "twenty pounds" lands wrong.
    "en_GB-alba-medium": "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
    "en_GB-alan-medium": "en/en_GB/alan/medium/en_GB-alan-medium.onnx",
    "en_GB-northern_english_male-medium": (
        "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"
    ),
    # German. Thorsten is the best-supported German voice in this collection;
    # the low variant is here to measure what the quality tier actually costs.
    "de_DE-thorsten-medium": "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "de_DE-thorsten-low": "de/de_DE/thorsten/low/de_DE-thorsten-low.onnx",
    "de_DE-kerstin-low": "de/de_DE/kerstin/low/de_DE-kerstin-low.onnx",
}


def download(url: str, destination: Path) -> None:
    """Fetch a URL to disk, reporting progress on a single line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"  already present: {destination.name}")
        return

    print(f"  fetching {destination.name} ...", end="", flush=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    # Rename only on success so an interrupted download is never mistaken for a
    # complete one on the next run.
    partial.rename(destination)
    print(f" {destination.stat().st_size / 1_000_000:.1f} MB")


def fetch_piper(target: Path) -> Path | None:
    """Download and unpack the Piper binary for this platform."""
    key = (platform.system(), platform.machine())
    asset = PIPER_ASSETS.get(key)
    if asset is None:
        print(f"No prebuilt Piper for {key}. Install it manually and set VA_PIPER_BIN.")
        return None

    binary = target / ("piper.exe" if key[0] == "Windows" else "piper")
    if binary.exists():
        print(f"  already present: {binary.name}")
        return binary

    archive = target / asset
    download(f"{PIPER_RELEASES}/{asset}", archive)

    print("  unpacking ...")
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target.parent)
    else:
        shutil.unpack_archive(str(archive), str(target.parent))
    archive.unlink(missing_ok=True)

    if not binary.exists():
        print(f"Unpacked archive did not contain {binary}", file=sys.stderr)
        return None
    binary.chmod(0o755)
    return binary


def fetch_voice(name: str, voices_dir: Path) -> Path | None:
    """Download a voice model and its config."""
    relative = VOICES.get(name)
    if relative is None:
        print(f"Unknown voice {name!r}. Choose from: {', '.join(VOICES)}")
        return None

    model = voices_dir / f"{name}.onnx"
    download(f"{VOICES_BASE}/{relative}", model)
    # Piper refuses to start without the sidecar config.
    download(f"{VOICES_BASE}/{relative}.json", voices_dir / f"{name}.onnx.json")
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice",
        action="append",
        choices=[*sorted(VOICES), "all"],
        help=(
            "voice to fetch; repeat for several, or pass 'all' to fetch every "
            "candidate for bench/voices.py (default: en_GB-alba-medium)"
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("models") / "piper",
        help="where to install (gitignored by default)",
    )
    args = parser.parse_args(argv)

    target = args.dest.resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f"Piper -> {target}")
    binary = fetch_piper(target)

    wanted = args.voice or ["en_GB-alba-medium"]
    if "all" in wanted:
        wanted = sorted(VOICES)

    print(f"Voices -> {target / 'voices'}")
    fetched = [fetch_voice(name, target / "voices") for name in wanted]
    voice = next((v for v in fetched if v is not None), None)

    if binary is None or any(v is None for v in fetched):
        return 1

    print()
    print("Done. Settings discovers these under models/piper; to override, set:")
    print(f"  VA_PIPER_BIN={binary}")
    print(f"  VA_PIPER_VOICE={voice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
