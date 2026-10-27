"""A saved cycle selection must not silently launch a different guest."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from run_selected_prefill import verify_guest_digest, verify_library_digest


class SelectedGuestTest(unittest.TestCase):
    def test_changed_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'libgolem.so'
            p.write_bytes(b'measured library')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            verify_library_digest(p,digest)
            p.write_bytes(b'new library')
            with self.assertRaisesRegex(ValueError,'Selected library changed'):
                verify_library_digest(p,digest)

    def test_changed_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'guest'
            p.write_bytes(b'measured guest')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            verify_guest_digest(p,digest)
            p.write_bytes(b'new guest')
            with self.assertRaisesRegex(ValueError,'Selected guest changed'):
                verify_guest_digest(p,digest)


if __name__ == '__main__':
    unittest.main()
