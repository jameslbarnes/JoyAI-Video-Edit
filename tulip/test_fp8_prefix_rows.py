import unittest
from xvideo.models.dit.prefix_attention import protected_text_rows


class ProtectedTextRows(unittest.TestCase):
    def test_every_length_preserves_the_complete_query_quantization_block(self):
        for prefix in (0, 12, 2592, 2688):
            for capacity in (512, 640, 1024):
                for tokens in range(1, capacity+1):
                    active = protected_text_rows(prefix, tokens, capacity)
                    self.assertGreaterEqual(active, tokens)
                    self.assertLessEqual(active, min(capacity,tokens+127))
                    if active < capacity:
                        self.assertEqual((prefix+active)%128,0)
                    # The real queries' final quantization block retains every
                    # padded query that participated in its original scale.
                    original_end = min(prefix+capacity, ((prefix+tokens+127)//128)*128)
                    self.assertEqual(prefix+active, original_end)

    def test_648p_examples_and_invalid_lengths(self):
        self.assertEqual(protected_text_rows(2592,517,1024),608)
        self.assertEqual(protected_text_rows(2592,497,512),512)
        self.assertEqual(protected_text_rows(2592,1024,1024),1024)
        for args in ((-1,5,512),(0,0,512),(0,513,512)):
            with self.assertRaises(ValueError): protected_text_rows(*args)

if __name__ == '__main__': unittest.main()
