from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "notes_to_pdf.py"


class NotesToPdfTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_generate_check_and_sync_track_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notes = Path(directory) / "notes.md"
            pdf = Path(directory) / "notes.pdf"
            original = "# Demo paper\n\nA short paragraph with **Markdown**.\n\n- First point\n"
            notes.write_text(original, encoding="utf-8")

            generated = self.run_cli("generate", str(notes))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(pdf.is_file())
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-1.4"))
            expected_hash = hashlib.sha256(original.encode()).hexdigest().encode()
            self.assertIn(b"/NotesSHA256 (" + expected_hash + b")", pdf.read_bytes())
            self.assertEqual(self.run_cli("check", str(notes)).returncode, 0)

            notes.write_text(original + "\n## New result\n\nUpdated notes.\n", encoding="utf-8")
            stale = self.run_cli("check", str(notes))
            self.assertEqual(stale.returncode, 1)
            self.assertIn("STALE", stale.stderr)

            synced = self.run_cli("sync", str(notes))
            self.assertEqual(synced.returncode, 0, synced.stderr)
            self.assertEqual(self.run_cli("check", str(notes)).returncode, 0)
            self.assertEqual(notes.read_text(encoding="utf-8"), original + "\n## New result\n\nUpdated notes.\n")

    def test_custom_output_and_missing_pdf_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            notes = Path(directory) / "notes.md"
            output = Path(directory) / "exports" / "paper.pdf"
            notes.write_text("# Title\n", encoding="utf-8")
            missing = self.run_cli("check", str(notes), "--output", str(output))
            self.assertEqual(missing.returncode, 1)
            generated = self.run_cli("generate", str(notes), "--output", str(output))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(output.is_file())
            self.assertEqual(self.run_cli("check", str(notes), "--output", str(output)).returncode, 0)


if __name__ == "__main__":
    unittest.main()
