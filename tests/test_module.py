# This file is part of Tryton.  The COPYRIGHT file at the top level of
# this repository contains the full copyright notices and license terms.
import unittest
from unittest.mock import MagicMock, patch

from trytond.modules.kbindex.kb import KbIndex


class KbIndexTestCase(unittest.TestCase):
    'Test KbIndex module'

    def test_sanitize_tsquery_content(self):
        self.assertEqual(
            KbIndex._sanitize_tsquery_content('foo\\nbar_baz  qux'),
            'foo | bar | baz | qux')

    def test_search_text_defaults_null_weight(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [(1, 0.5)]

        with patch.object(KbIndex, 'search_check'), \
                patch.object(KbIndex, 'get_cursor', return_value=cursor), \
                patch.object(
                    KbIndex, 'search_vars_to_query',
                    side_effect=lambda table, query, *args: query):
            result = KbIndex.search_text('hello', language='en')

        query, parameters = cursor.execute.call_args.args
        self.assertIn('COALESCE', query)
        self.assertEqual(parameters[0], 'D')
        self.assertEqual(result, {1: 0.5})
