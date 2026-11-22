import struct
import tempfile
from pathlib import Path
import unittest
from scripts.review_physmani_settings_dat_r10 import MARKER, parse_crash_marker, inspect_exact_file


def encoded(value=b'No', key=MARKER):
    return struct.pack('<iB', 1, 1)+struct.pack('<i', len(key))+key+struct.pack('<i',len(value))+value+b'\x00'


class ExactCrashMarkerReview(unittest.TestCase):
    def test_only_two_known_values_and_exact_bytes(self):
        for value in (b'No', b'Yes'):
            payload=encoded(value)
            self.assertEqual(parse_crash_marker(payload)['value'], value.decode())
            self.assertEqual(parse_crash_marker(payload)['hex'], payload.hex())

    def test_unknown_key_value_structure_or_extra_setting_rejected(self):
        payload=encoded()
        invalid=[encoded(key=b'UNKNOWN_CRASH_MARKER_001'), encoded(b'no'), encoded(b'On'),
                 struct.pack('<i',2)+payload[4:], b'\x00'+payload[1:],
                 payload[:4]+b'\x02'+payload[5:], payload[:-1], payload+b'\x00',
                 payload[:-1]+encoded(key=b'realPhysicsSetting'),
                 payload[:5]+struct.pack('<i',-1)+payload[9:]]
        for data in invalid:
            with self.subTest(data=data.hex()), self.assertRaises(ValueError):
                parse_crash_marker(data)

    def test_unknown_file_and_real_configuration_not_exempted(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'settings.dat';marker.write_bytes(encoded())
            unrelated=Path(directory)/'usrset.txt';unrelated.write_bytes(encoded())
            original=marker.read_bytes()
            self.assertTrue(inspect_exact_file(marker,marker)['read_only'])
            with self.assertRaises(ValueError):
                inspect_exact_file(unrelated,marker)
            marker.write_bytes(original+b'physicsSetting=changed')
            with self.assertRaises(ValueError):
                inspect_exact_file(marker,marker)
            self.assertEqual(marker.read_bytes(),original+b'physicsSetting=changed')


if __name__ == '__main__':
    unittest.main()
