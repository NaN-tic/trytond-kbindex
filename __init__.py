# This file is part of Tryton.  The COPYRIGHT file at the top level of
# this repository contains the full copyright notices and license terms.
from trytond.pool import Pool

from . import kb


def register():
    Pool.register(
        kb.KbIndex,
        kb.KbReindexQuestion,
        kb.KbReindexResult,
        module='kbindex', type_='model')
    Pool.register(
        kb.KbReindex,
        module='kbindex', type_='wizard')
