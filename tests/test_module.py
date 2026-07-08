# This file is part of Tryton.  The COPYRIGHT file at the top level of
# this repository contains the full copyright notices and license terms.
import unittest

from trytond.modules.kbindex.kb import KbIndex


class KbIndexTestCase(unittest.TestCase):
    'Test KbIndex module'

    def test_sanitize_tsquery_content(self):
        self.assertEqual(
            KbIndex._sanitize_tsquery_content('foo\\nbar_baz  qux'),
            'foo | bar | baz | qux')
