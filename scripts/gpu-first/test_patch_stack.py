"""Regression test for ordered, overlapping native patches (no downloads)."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(argv, cwd, check=True):
    return subprocess.run(argv, cwd=cwd, check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", default=shutil.which("cmake"))
    args = parser.parse_args()
    if not args.cmake:
        parser.error("CMake must already be installed")
    repo = Path(__file__).resolve().parents[2]
    scratch = repo / ".build" / "patch-stack-tests"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="owned-fixture-", dir=scratch) as folder:
        root = Path(folder).resolve()
        assert root.is_relative_to(scratch.resolve())
        source, patches = root / "source", root / "patches"
        source.mkdir()
        patches.mkdir()
        run(["git", "init", "-q"], source)
        # Local fixture configuration only; do not change the user's Git config.
        run(["git", "config", "user.name", "Patch regression fixture"], source)
        run(["git", "config", "user.email", "fixture@localhost"], source)
        run(["git", "config", "core.autocrlf", "false"], source)
        target = source / "value.txt"
        versions = ["before\n", "one\n", "two\n", "three\n"]
        target.write_text(versions[0], encoding="utf8")
        run(["git", "add", "."], source)
        run(["git", "commit", "-qm", "fixture base"], source)
        for idx, content in enumerate(versions[1:], 1):
            target.write_text(content, encoding="utf8")
            run(["git", "diff", "--binary", f"--output={patches / f'{idx:03d}.patch'}"], source)
            run(["git", "add", "."], source)
            run(["git", "commit", "-qm", f"fixture {idx}"], source)
        applier = [args.cmake, f"-DPATCH_DIR={patches}", "-DPATCH_LABEL=fixture",
                   "-P", str(repo / "cmake" / "apply-git-patches.cmake")]
        for prefix in range(len(versions)):
            target.write_text(versions[prefix], encoding="utf8")
            before_index = (source / ".git/index").read_bytes()
            run(applier, source)
            assert target.read_text(encoding="utf8") == versions[-1], prefix
            assert (source / ".git/index").read_bytes() == before_index, "real index modified"
            run(applier, source)
            assert target.read_text(encoding="utf8") == versions[-1], "reconfigure changed content"
            assert not list(source.glob(".ollama-patch-check-*")), "probe directory leaked"
        target.write_text("unrelated conflicting content\n", encoding="utf8")
        result = run(applier, source, check=False)
        assert result.returncode != 0, "conflicting edits were accepted"
        assert target.read_text(encoding="utf8") == "unrelated conflicting content\n", "conflict damaged source"
        assert not list(source.glob(".ollama-patch-check-*")), "failed probe directory leaked"
    print("PASS: clean, all partial prefixes, repeated apply, index preservation, conflict preservation")


if __name__ == "__main__":
    main()
