import json
import logging
import re
from collections import defaultdict

import psycopg
import tiktoken
from pgvector.psycopg import register_vector
from sql import Asc, Cast, Desc, Literal
from sql.functions import CharLength, Function, Position, Substring
from sql.operators import (
    Add, BinaryOperator, Equal, GreaterEqual, In, LessEqual, Or, Sub)
from trytond import backend
from trytond.exceptions import UserError
from trytond.i18n import gettext
from trytond.model import Index, ModelSQL, ModelView, fields
from trytond.modules.widgets.vector import Vector
from trytond.pool import Pool
from trytond.tools import grouped_slice
from trytond.transaction import Transaction
from trytond.wizard import Button, StateTransition, StateView, Wizard

EMBEDDINGS_DIMENSION = 1536
TOKEN_LIMIT = 8192
logger = logging.getLogger(__name__)


class MatchTextSearch(BinaryOperator):
    __slots__ = ()
    _operator = '@@'


class CosineDistance(BinaryOperator):
    __slots__ = ()
    _operator = '<=>'


class RankTextSearch(Function):
    __slots__ = ()
    _function = 'TS_RANK_CD'


class FTSVector(Function):
    __slots__ = ()
    _function = 'FTS_VECTOR'


class FTSQuery(Function):
    __slots__ = ()
    _function = 'FTS_QUERY'


class SetWeight(Function):
    __slots__ = ()
    _function = 'SETWEIGHT'


class Similarity(Function):
    __slots__ = ()
    _function = 'similarity'


class KbIndex(ModelSQL, ModelView):
    "Index"
    __name__ = 'kb.index'

    text = fields.Text('Text', required=True, readonly=True)
    embedding = Vector('Embedding', size=EMBEDDINGS_DIMENSION, readonly=True)
    resource = fields.Reference('Resource', selection='get_resources',
        required=True, readonly=True)
    weight = fields.Char("Weight",
        help="Priority of the index record. 'A' for high priority, "
        "'D' for default.")
    language_code = fields.Selection([
            ('es', 'Spanish'),
            ('ca', 'Catalan'),
            ('en', 'English'),
            ], 'Language Code', required=True, readonly=True)

    @classmethod
    def __setup__(cls):
        super().__setup__()
        table = cls.__table__()
        cls._sql_indexes.update({
            Index(table, (table.resource, Index.Equality())),
            Index(table, (
                Substring(table.resource, 1, Position(',', table.resource) - 1),
                Index.Equality()
            )),
        })

    @classmethod
    def __register__(cls, module_name):
        if backend.name == 'postgresql':
            cursor = cls.get_cursor()

        table = cls.__table_handler__(module_name)
        if table.column_exist('reference') and not table.column_exist('resource'):
            table.column_rename('reference', 'resource')

        super().__register__(module_name)
        if backend.name != 'postgresql':
            return

        cursor.execute("""
            CREATE OR REPLACE FUNCTION fts_vector(language varchar, txt text)
            RETURNS tsvector AS $$
            BEGIN
                RETURN to_tsvector(
                    CASE language
                        WHEN 'en' THEN 'english'
                        WHEN 'ca' THEN 'catalan'
                        WHEN 'es' THEN 'spanish'
                        ELSE 'simple'
                    END::regconfig,
                    txt
                );
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """)
        cursor.execute("""
            CREATE OR REPLACE FUNCTION fts_query(language varchar, txt text)
            RETURNS tsquery AS $$
            BEGIN
                RETURN to_tsquery(
                    CASE language
                        WHEN 'en' THEN 'english'
                        WHEN 'ca' THEN 'catalan'
                        WHEN 'es' THEN 'spanish'
                        ELSE 'simple'
                    END::regconfig,
                    txt
                );
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """)
        try:
            cursor.execute('SAVEPOINT create_index;')
            cursor.execute(f"""
                CREATE INDEX kb_index_fts
                ON {cls._table}
                USING GIN (fts_vector(language_code, text));
            """)
        except psycopg.errors.DuplicateTable:
            cursor.execute('ROLLBACK TO SAVEPOINT create_index;')
        finally:
            cursor.execute('RELEASE SAVEPOINT create_index;')

    @classmethod
    def search_rec_name(cls, _name, clause):
        _, _operator, operand, *_extras = clause
        if operand is None:
            return []
        if operand.startswith('%') and operand.endswith('%'):
            operand = operand[1:-1]

        results = cls.search_text(operand) or {}
        ids = set(results.keys())
        return [('id', 'in', ids)]

    @classmethod
    def copy(cls, records, default=None):
        if records:
            raise UserError(gettext('kbindex.msg_index_copy_forbidden'))
        return super().copy(records, default)

    @classmethod
    def create(cls, vlist):
        to_queue, vlist = [], [x.copy() for x in vlist]

        for index, values in enumerate(vlist):
            needs_embedding = cls.calculate_fields(values)
            if needs_embedding:
                to_queue.append(index)

        records = super().create(vlist)
        cls.__queue__.compute([records[index] for index in to_queue])
        return records

    @classmethod
    def write(cls, *args):
        to_queue, new_args = [], []
        actions = iter(args)

        for records, values in zip(actions, actions):
            values = values.copy()
            new_args += [records, values]
            needs_embedding = cls.calculate_fields(values)
            if needs_embedding:
                to_queue += records

        cls.__queue__.compute(to_queue)
        return super().write(*new_args)

    @classmethod
    def calculate_fields(cls, dictionary):
        if 'text' not in dictionary:
            return
        if not dictionary['text']:
            dictionary.setdefault('embedding', None)
        elif 'embedding' not in dictionary:
            return True

    @staticmethod
    def get_cursor():
        connection = Transaction().connection
        try:
            register_vector(connection)
        except psycopg.ProgrammingError:
            logger.exception(
                "Vector type not found on DB. Please install PostgreSQL's "
                "'pg_vector'. TIP: https://github.com/pgvector/pgvector")
        return connection.cursor()

    @classmethod
    def _check_domain_fields(cls, domain, ref_class):
        for condition in domain:
            if isinstance(condition, list):
                cls._check_domain_fields(condition, ref_class)
            elif isinstance(condition, tuple):
                field_name = condition[0].split('.')[0]
                assert field_name in ref_class._fields

    @classmethod
    def search_vars_to_query(cls, table, query, ref_filters, language, limit):
        if language:
            query.where &= table.language_code == language
        if ref_filters:
            query.where &= cls._process_resource_filters(ref_filters, table)
        if limit:
            query.limit = limit
        return query

    @classmethod
    def _process_resource_filters(cls, ref_filters, table):
        sql_converted_filters = []
        for model_name, domain in ref_filters.items():
            delimiter_position = Position(',', table.resource)
            resource_model = Substring(
                table.resource, 1, delimiter_position - 1)
            condition = Equal(resource_model, Literal(model_name))
            if domain:
                filter_table = Pool().get(model_name)
                subquery = filter_table.search(domain=domain, query=True)
                resource_id = Cast(
                    Substring(
                        table.resource,
                        Add(delimiter_position, 1),
                        Sub(CharLength(table.resource),
                            Add(CharLength(resource_model), 1))),
                    'INTEGER')
                condition &= In(resource_id, subquery)
            sql_converted_filters.append(condition)

        return Or(sql_converted_filters)

    @classmethod
    def search_check(cls, content, filters, language, limit):
        if backend.name != 'postgresql':
            raise AssertionError('This only works with PostgreSQL backend')

        assert isinstance(limit, int) and limit >= 0

        if language and language.strip():
            assert language in [_[0] for _ in cls.language_code.selection]

        if isinstance(filters, dict):
            for ref, domain in filters.items():
                assert ref in cls._get_resources()
                ref_class = Pool().get(ref)
                cls._check_domain_fields(domain, ref_class)

        if isinstance(content, str):
            assert content.strip()
        elif isinstance(content, list):
            assert len(content) == EMBEDDINGS_DIMENSION, (
                'Embedding must have '
                f'{EMBEDDINGS_DIMENSION} dimensions. Got {len(content)}.')
        else:
            raise AssertionError('Content must be a string or a list of floats')

    @classmethod
    def search_similarity(cls, content, ref_filters=None, language=None,
            threshold=0, limit=0):
        cls.search_check(content, ref_filters, language, limit)
        assert 0 <= threshold <= 1

        table = cls.__table__()
        similarity = Similarity(content, table.text)
        score = similarity.as_('score')

        select = table.select(table.id, score)
        select.where = GreaterEqual(similarity, threshold)
        select.order_by = Desc(score)
        select = cls.search_vars_to_query(
            table, select, ref_filters, language, limit)

        cursor = cls.get_cursor()
        cursor.execute(*select)
        return dict(cursor.fetchall())

    @classmethod
    def search_ilike(cls, content, ref_filters=None, language=None, limit=0,
            word_len=0, weights=None):
        cls.search_check(content, ref_filters, language, limit)
        table = cls.__table__()
        words = [
            w.strip() for w in content.strip().split() if len(w.strip()) > word_len]
        if not words:
            return {}

        ilike_clauses = [table.text.ilike(f'%{word}%') for word in words]
        condition = ilike_clauses[0]
        for clause in ilike_clauses[1:]:
            condition |= clause

        select = table.select(table.id, table.text)
        select.where = condition
        if weights:
            select.where &= table.weight.in_(weights)
        select = cls.search_vars_to_query(
            table, select, ref_filters, language, limit)

        cursor = cls.get_cursor()
        cursor.execute(*select)

        results = {}
        for record_id, text in cursor.fetchall():
            count = sum(word.lower() in text.lower() for word in words)
            score = count / len(words)
            results[record_id] = score

        res = dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True))
        if limit > 0 and limit > len(res):
            res = dict(list(res.items())[:limit])
        return res

    @staticmethod
    def _sanitize_tsquery_content(content):
        content = re.sub(r'\\[nrt]', ' ', content)
        return ' | '.join(re.findall(r'[^\W_]+', content, flags=re.UNICODE))

    @classmethod
    def search_text(cls, content, ref_filters: dict[str:list[tuple]] = None,
            language=None, limit=0, weights=None):
        content = content.strip()
        cls.search_check(content, ref_filters, language, limit)
        if language is None:
            results = {}
            for language_code, _ in cls.language_code.selection:
                results.update(cls.search_text(
                        content, ref_filters, language_code,
                        weights=weights))
            results = dict(sorted(
                    results.items(), key=lambda item: item[1], reverse=True))
            if limit:
                results = dict(list(results.items())[:limit])
            return results
        query_content = cls._sanitize_tsquery_content(content)
        if not query_content:
            return {}
        table = cls.__table__()
        query = table.select()

        base_vector = FTSVector(table.language_code, table.text)
        weight_as_char = Cast(table.weight, '\"char\"')
        weighted_vector = SetWeight(base_vector, weight_as_char)
        fts_query = FTSQuery(language or table.language_code, query_content)
        rank = RankTextSearch(weighted_vector, fts_query, 1).as_('rank')

        query.columns = [table.id.as_('id'), rank]
        query.where = MatchTextSearch(base_vector, fts_query)
        if weights:
            query.where &= table.weight.in_(weights)
        query.order_by = Desc(rank)
        query = cls.search_vars_to_query(
            table, query, ref_filters, language, limit)

        cursor = cls.get_cursor()
        cursor.execute(*query)
        results = dict(cursor.fetchall())

        if results:
            return results
        if len(content.strip()) < 4:
            return {}
        return cls.search_ilike(content, ref_filters, language,
            limit=limit, word_len=3, weights=weights)

    @classmethod
    def search_resource_ids(cls, content, model, domain=None, language=None,
            limit=0, weights=None):
        """Return resource IDs whose indexed text matches *content*."""
        domain = domain or []
        if backend.name == 'postgresql':
            index_scores = cls.search_text(
                content, {model: domain}, language=language, weights=weights)
            index_ids = list(index_scores)
        else:
            Resource = Pool().get(model)
            resources = Resource.search(domain)
            references = [str(resource) for resource in resources]
            if not references:
                return []
            index_domain = [
                    ('text', 'ilike', f'%{content.strip()}%'),
                    ('resource', 'in', references),
                    ]
            if weights:
                index_domain.append(('weight', 'in', weights))
            indexes = cls.search(index_domain)
            index_ids = [index.id for index in indexes]

        if not index_ids:
            return []
        table = cls.__table__()
        cursor = Transaction().connection.cursor()
        references = {}
        for sub_ids in grouped_slice(index_ids):
            where = table.id.in_(list(sub_ids))
            if weights:
                where &= table.weight.in_(weights)
            cursor.execute(*table.select(
                    table.id, table.resource,
                    where=where))
            references.update(cursor.fetchall())

        prefix = f'{model},'
        resource_ids = []
        seen = set()
        for index_id in index_ids:
            reference = references.get(index_id, '')
            if not reference.startswith(prefix):
                continue
            try:
                resource_id = int(reference[len(prefix):])
            except ValueError:
                continue
            if resource_id not in seen:
                resource_ids.append(resource_id)
                seen.add(resource_id)
                if limit and len(resource_ids) >= limit:
                    break
        return resource_ids

    @classmethod
    def search_embedding(cls, content, origin, threshold=1,
            ref_filters: dict[str:list[tuple]] = None, language=None, limit=0):
        cls.search_check(content, ref_filters, language, limit)
        assert 0 <= threshold <= 2
        if isinstance(content, str):
            embedding = cls.compute_embedding(content, origin)
        else:
            embedding = content

        table = cls.__table__()
        query = table.select()

        distance = CosineDistance(table.embedding, Cast(embedding, 'VECTOR'))
        score = distance.as_('score')

        query.columns = [table.id.as_('id'), score]
        query.where = LessEqual(distance, threshold)
        query.order_by = Asc(score)
        query = cls.search_vars_to_query(
            table, query, ref_filters, language, limit)

        cursor = cls.get_cursor()
        cursor.execute(*query)
        return dict(cursor.fetchall())

    @classmethod
    def combine_searches(cls, text_results=None, embedding_results=None):
        results = defaultdict(int)

        if text_results:
            xmax = max(text_results.values()) or 1
            for index_id, score in text_results.items():
                results[index_id] = score / xmax

        if embedding_results:
            for index_id, score in embedding_results.items():
                results[index_id] += 1 - (score / 2)

        return dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True))

    @classmethod
    def _get_resources(cls):
        return []

    @classmethod
    def get_resources(cls):
        Model = Pool().get('ir.model')
        models = Model.search([('name', 'in', cls._get_resources())])
        return [(model.name, model.string) for model in models]

    @classmethod
    def compute_embedding(cls, text, origin):
        try:
            Configuration = Pool().get('ai.configuration')
        except KeyError:
            logger.error("ai.configuration is not available.")
            return

        if not text or not text.strip():
            return
        configuration = Configuration(1)
        embedding_model = configuration.get_default_embedding_model()
        if not embedding_model.get_client():
            logger.error("OpenRouter is not configured.")
            return

        try:
            encoding = tiktoken.encoding_for_model(
                embedding_model.model_name.split('/')[-1])
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        tokens = encoding.encode(text)
        if len(tokens) > TOKEN_LIMIT:
            logger.warning(
                f'Text exceeds the maximum token limit of {TOKEN_LIMIT}. '
                f'Truncating text to the first {TOKEN_LIMIT} tokens '
                f'(from {len(tokens)}).')
            text = encoding.decode(tokens[:TOKEN_LIMIT])

        response = embedding_model.get_embeddings(
            text, origin, encoding_format='float',
            dimensions=EMBEDDINGS_DIMENSION,
            )
        return response.data[0].embedding

    @classmethod
    def compute(cls, records):
        for record in records:
            record.embedding = record.compute_embedding(record.text, record)
        cls.save(records)

    @classmethod
    def compute_indexes(cls, dictionary):
        old_indexes = {
            (str(index.resource), index.text): index
            for index in cls.search([('resource', 'in', dictionary.keys())])
        }
        to_save = []
        to_delete = old_indexes.copy()
        for indexes in dictionary.values():
            for index in indexes:
                key = (str(index.resource), index.text)
                if key not in old_indexes:
                    to_save.append(index)
                else:
                    to_delete.pop(key, None)

        cls.save(to_save)
        cls.delete(to_delete.values())


class KbReindexQuestion(ModelView):
    'Reindex Question'
    __name__ = 'kb.reindex.question'

    models = fields.MultiSelection(string="Models", selection="get_models")

    @staticmethod
    def default_models():
        Indexation = Pool().get('kb.index')
        return Indexation._get_resources()

    @classmethod
    def get_models(cls):
        pool = Pool()
        Indexation = pool.get('kb.index')

        return [
            (model, name)
            for model, name in Indexation.get_resources()
            if getattr(pool.get(model), 'indexate', False)
        ]


class KbReindexResult(ModelView):
    'Reindex Result'
    __name__ = 'kb.reindex.result'

    result = fields.Text('Result', readonly=True)


class KbReindex(Wizard):
    'Reindex'
    __name__ = 'kb.reindex'
    start_state = 'question'

    question = StateView('kb.reindex.question',
        'kbindex.kb_reindex_question_form_view', [
            Button('Cancel', 'end', 'tryton-cancel'),
            Button('Indexate', 'indexate', 'tryton-ok', default=True),
        ])
    indexate = StateTransition()
    result = StateView('kb.reindex.result',
        'kbindex.kb_reindex_result_form_view', [
            Button('Close', 'end', 'tryton-close'),
        ])

    def transition_indexate(self):
        if len(self.question.models) == 0:
            return 'question'
        pool = Pool()

        comments = []
        cursor = Transaction().connection.cursor()
        savepoint = 'wizard_indexate'
        for model in self.question.models:
            try:
                cursor.execute(f'SAVEPOINT {savepoint};')
                Model = pool.get(model)
                for records in grouped_slice(Model.search([])):
                    Model.indexate(records)
                comments.append({'on': model, 'succeed': True})
            except Exception as exception:
                cursor.execute(f'ROLLBACK TO SAVEPOINT {savepoint};')
                comments.append({'on': model, 'error': str(exception)})
            finally:
                cursor.execute(f'RELEASE {savepoint};')

        self.result.result = json.dumps(comments, indent=2)
        return 'result'

    def default_result(self, _fields):
        return {
            'result': self.result.result or '',
        }
