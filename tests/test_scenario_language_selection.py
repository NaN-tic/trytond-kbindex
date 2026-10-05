import unittest

from proteus import Model
from trytond.tests.test_tryton import drop_db
from trytond.tests.tools import activate_modules


class TestLanguageSelection(unittest.TestCase):

    def setUp(self):
        drop_db()
        super().setUp()

    def tearDown(self):
        drop_db()
        super().tearDown()

    def test(self):
        activate_modules('kbindex')

        Language = Model.get('ir.lang')
        english, = Language.find([('code', '=', 'en')])
        custom, = english.duplicate({
                'code': 'x-test',
                'name': 'Test Language',
                'translatable': False,
                'pg_text_search': None,
                })

        KbIndex = Model.get('kb.index')
        languages = dict(KbIndex.get_languages(context={}))

        self.assertEqual(languages[custom.code], custom.name)
        self.assertEqual(
            languages,
            {language.code: language.name for language in Language.find([])})
