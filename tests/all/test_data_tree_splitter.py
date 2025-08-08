"""Module to add unit test for DataTreeSplitter class."""
import unittest

import pandas as pd

from simplerag.document_splitters.hierarchical import DataTreeSplitter
from simplerag.document_splitters.hierarchical import DocNode

class TestDataTreeSplitter(unittest.TestCase):
    """Class to test the hierarchical spliter based on data."""

    def test_same_block_in_secuence_of_titles(self):
        """
        Test that when multiple centered lines are together all have the same
        block.
        """
        # Reglamento del personal académico (Pag 30)
        data = pd.DataFrame([
            {"page": 0, "text": "SECCIÓN",
                "left": 0.439264903630659,
                "top": 0.128091872791519,
                "right": 0.515015688032273,
                "bottom": 0.139796819787986,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "SEGUNDA",
                "left": 0.522037950097116,
                "top": 0.128091872791519,
                "right": 0.605109816225908,
                "bottom": 0.139686395759717,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "INGRESO",
                "left": 0.320633497684148,
                "top": 0.148299469964664,
                "right": 0.396533691916928,
                "bottom": 0.159673144876325,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "PROFESORAS",
                "left": 0.616913192888092,
                "top": 0.151943462897527,
                "right": 0.722695353354251,
                "bottom": 0.159673144876325,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert "block" in splitter.data
        assert list(splitter.data[splitter.data['block'] == 0].index) == [0, 1, 2, 3]

    def test_same_block_in_secuence_of_titles_with_lowercase(self):
        """
        Test that when multiple centered lines are together and the seccond
        is all lowercace (hence smaller line height), all have the same
        block.
        """
        # Reglamento del personal académico (Pag 30)
        data = pd.DataFrame([
            {"page": 0, "text": "INGRESO",
                "left": 0.320633497684148,
                "top": 0.148299469964664,
                "right": 0.396533691916928,
                "bottom": 0.159673144876325,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "PROFESORAS",
                "left": 0.616913192888092,
                "top": 0.151943462897527,
                "right": 0.722695353354251,
                "bottom": 0.159673144876325,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "Y",
                "left": 0.351710742566861,
                "top": 0.171709363957597,
                "right": 0.362169430748543,
                "bottom": 0.179439045936396,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "PARCIAL",
                "left": 0.620797848498431,
                "top": 0.171930212014134,
                "right": 0.691468698640371,
                "bottom": 0.179659893992933,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
        ])

        splitter = DataTreeSplitter(data, loader='mixed')
        splitter.analyze()

        assert "block" in splitter.data
        assert list(splitter.data[splitter.data['block'] == 0].index) == [0, 1, 2, 3]

    def test_same_block_in_secuence_of_lines(self):
        """
        Test that when multiple lines are together all have the same block.
        """
        # Reglamento del personal académico (Pag 30)
        data = pd.DataFrame([
            {"page": 0, "text": "Artículo",
                "left": 0.139796819787986,
                "top": 0.306868374558304,
                "right": 0.214104288062154,
                "bottom": 0.319125441696113,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "63",
                "left": 0.221126550126998,
                "top": 0.308193462897527,
                "right": 0.240848647841028,
                "bottom": 0.322106890459364,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "la",
                "left": 0.147616913192888,
                "top": 0.326855123674912,
                "right": 0.162109666816077,
                "bottom": 0.339112190812721,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "División",
                "left": 0.168982519049753,
                "top": 0.32773851590106,
                "right": 0.243089795308531,
                "bottom": 0.339112190812721,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert "block" in splitter.data
        assert list(splitter.data[splitter.data['block'] == 0].index) == [0, 1, 2, 3]

    def test_different_block_for_separated_lines(self):
        """
        Test that when text has a vertical separation it's assigned two blocks.
        """
        # Ley Organica (Pag 19)
        data = pd.DataFrame([
            {"page": 0, "text": "LEY",
                "left": 0.184670551322277,
                "top": 0.746466431095406,
                "right": 0.223965336919169,
                "bottom": 0.75761925795053,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "GUANAJUATO",
                "left": 0.623935454952936,
                "top": 0.746024734982332,
                "right": 0.771402958314657,
                "bottom": 0.761042402826855,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "Guanajuato,",
                "left": 0.309875989840132,
                "top": 0.786108657243816,
                "right": 0.417152248617959,
                "bottom": 0.802009717314488,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "2007",
                "left": 0.603466308083072,
                "top": 0.790083922261484,
                "right": 0.646944568952637,
                "bottom": 0.800795053003534,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert "block" in splitter.data
        assert splitter.data.loc[0, 'block'] == splitter.data.loc[1, 'block']
        assert splitter.data.loc[1, 'block'] != splitter.data.loc[2, 'block']
        assert splitter.data.loc[2, 'block'] == splitter.data.loc[3, 'block']

    def test_different_block_for_separated_lines_in_content(self):
        """
        Test that when text in the content of the document (not title) has a
        small vertical separation (between articles) it's assigned two blocks.
        """
        # Reglamento del personal académico (Pag 18)
        data = pd.DataFrame([
            {"page": 0, "text": "carse",
                "left": 0.547288211564321,
                "top": 0.67071554770318,
                "right": 0.589870013446885,
                "bottom": 0.678003533568905,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "español.",
                "left": 0.810847153742716,
                "top": 0.665746466431095,
                "right": 0.880472135066488,
                "bottom": 0.682089222614841,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "Análisis",
                "left": 0.607500373524578,
                "top": 0.706603356890459,
                "right": 0.661287912744659,
                "bottom": 0.717866607773852,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "del",
                "left": 0.814881219184222,
                "top": 0.706824204946997,
                "right": 0.834603316898252,
                "bottom": 0.717866607773852,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert "block" in splitter.data
        assert splitter.data.loc[0, 'block'] == splitter.data.loc[1, 'block']
        assert splitter.data.loc[1, 'block'] != splitter.data.loc[2, 'block']
        assert splitter.data.loc[2, 'block'] == splitter.data.loc[3, 'block']

    def test_set_document_title_as_lvl0_node(self):
        """
        Test that the document title (when is present in the first page alone)
        is set as lvl1 node.
        """
        # Reglamento del personal académico (Pag 1)
        data = pd.DataFrame([
            {"page": 0, "text": "REGLAMENTO",
                "left": 0.236216943074854,
                "top": 0.446775618374558,
                "right": 0.398326609890931,
                "bottom": 0.458701413427562,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "ACADÉMICO",
                "left": 0.578963095771702,
                "top": 0.444015017667845,
                "right": 0.71985656656208,
                "bottom": 0.458701413427562,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "DE",
                "left": 0.264007171671896,
                "top": 0.467314487632509,
                "right": 0.293889137905274,
                "bottom": 0.478577738515901,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 0, "text": "GUANAJUATO",
                "left": 0.540415359330644,
                "top": 0.466762367491166,
                "right": 0.692066337965038,
                "bottom": 0.481890459363958,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            # Dummy words to stablish document writable area
            {"page": 2, "text": "dummy_top_left",
                "left": 0.104138652323323,
                "top": 0.148299469964664,
                "right": 0.125952487673689,
                "bottom": 0.159673144876325,
            "line": 2, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "dummy_bottom_right",
                "left": 0.880770954728821,
                "top": 0.890348939929329,
                "right": 0.896160167339011,
                "bottom": 0.897416077738516,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert len(splitter.root.children) > 0
        assert isinstance(splitter.root.children[0], DocNode) is True
        assert splitter.root.children[0].name == 'REGLAMENTO ACADÉMICO DE GUANAJUATO'

    def test_set_normal_centered_text_as_lvl1_node(self):
        """
        Test that centered text with no particular content is set as lvl1 node.
        """
        # Reglamento del personal académico (Pag 3)
        data = pd.DataFrame([
            {"page": 2, "text": "EXPOSICIÓN",
                "left": 0.367847004332885,
                "top": 0.148299469964664,
                "right": 0.474525623786045,
                "bottom": 0.159673144876325,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "MOTIVOS",
                "left": 0.510085163603765,
                "top": 0.151833038869258,
                "right": 0.588226505304049,
                "bottom": 0.159673144876325,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            # Dummy words to stablish page writable area
            {"page": 2, "text": "La",
                "left": 0.104138652323323,
                "top": 0.188162544169611,
                "right": 0.125952487673689,
                "bottom": 0.199536219081272,
            "line": 2, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "la",
                "left": 0.838487972508591,
                "top": 0.905035335689046,
                "right": 0.852980726131779,
                "bottom": 0.917292402826855,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert len(splitter.root.children) > 0
        assert isinstance(splitter.root.children[0], DocNode) is True
        assert splitter.root.children[0].name == 'EXPOSICIÓN MOTIVOS'

    def test_set_bigger_text_as_lvl1_node(self):
        """
        Test that bigger text with no particular content is set as lvl1 node.
        """
        # Modelo educativo (Pag 4)
        data = pd.DataFrame([
            {"page": 1, "text": "1",
                "left": 0.0708235294117647,
                "top": 0.0573636363636364,
                "right": 0.0892941176470588,
                "bottom": 0.0754545454545455,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 1, "text": "Presentación",
                "left": 0.102235294117647,
                "top": 0.052,
                "right": 0.347176470588235,
                "bottom": 0.0754545454545455,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 1, "text": "Modelo",
                "left": 0.0709411764705882,
                "top": 0.102818181818182,
                "right": 0.0852941176470588,
                "bottom": 0.113090909090909,
            "line": 1, "column": 0, "col_position": 0, "group": 1},
            # Dummy words to stablish page writable area
            {"page": 1, "text": "objetivos",
                "left": 0.784705882352941,
                "top": 0.860636363636364,
                "right": 0.850352941176471,
                "bottom": 0.872,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert len(splitter.root.children) > 0
        assert isinstance(splitter.root.children[0], DocNode) is True
        assert splitter.root.children[0].name == '1 Presentación'

    def test_set_lvl2_titulo_as_lvl2_node(self):
        """
        Test that centered text that matches level 2 Regex (Título ...) is
        considered as a level2 node
        """
        # Reglamento del personal académico (Pag 9)
        data = pd.DataFrame([
            {"page": 2, "text": "REGLAMENTO",
                "left": 0.240848647841028,
                "top": 0.207597173144876,
                "right": 0.400119527864934,
                "bottom": 0.21952296819788,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "ACADÉMICO",
                "left": 0.576871358135365,
                "top": 0.204836572438163,
                "right": 0.715224861795906,
                "bottom": 0.21952296819788,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "TÍTULO",
                "left": 0.3998207082026,
                "top": 0.267336572438163,
                "right": 0.467951591214702,
                "bottom": 0.279262367491166,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "PRIMERO",
                "left": 0.475422082773047,
                "top": 0.267888692579505,
                "right": 0.555804571940834,
                "bottom": 0.279262367491166,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            # Dummy words to stablish page writable area
            {"page": 2, "text": "dummy_left",
                "left": 0.102046914686986,
                "top": 0.207597173144876,
                "right": 0.172269535335425,
                "bottom": 0.438825088339223,
            "line": 2, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "dummy_right",
                "left": 0.815628268340057,
                "top": 0.905918727915194,
                "right": 0.852084267144778,
                "bottom": 0.917292402826855,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        lvl1_node = splitter.root.children[0]
        assert len(lvl1_node.children) > 0
        assert isinstance(lvl1_node.children[0], DocNode) is True
        assert lvl1_node.children[0].name == 'TÍTULO PRIMERO'

    def test_set_lvl2_roman_as_lvl2_node(self):
        """
        Test that centered text that matches level 2 Regex (I. ...) is
        considered as a level2 node
        """
        # Ley Organica (Pag 3)
        data = pd.DataFrame([
            {"page": 2, "text": "DICTAMEN",
                "left": 0.397728970566263,
                "top": 0.506625441696113,
                "right": 0.558792768564172,
                "bottom": 0.518551236749117,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "I.",
                "left": 0.349768414761691,
                "top": 0.547040636042403,
                "right": 0.360525922605707,
                "bottom": 0.558414310954064,
            "line": 1, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "legislativo",
                "left": 0.496488868967578,
                "top": 0.550463780918728,
                "right": 0.606753324368743,
                "bottom": 0.558414310954064,
            "line": 1, "column": 0, "col_position": 0, "group": 1},
            # Dummy words to stablish page writable area
            {"page": 2, "text": "DICTAMEN",
                "left": 0.10428806215449,
                "top": 0.147747349823322,
                "right": 0.226057074555506,
                "bottom": 0.159673144876325,
            "line": 2, "column": 0, "col_position": 0, "group": 2},
            {"page": 2, "text": "de",
                "left": 0.832959808755416,
                "top": 0.865172261484099,
                "right": 0.852383086807112,
                "bottom": 0.877650176678445,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        lvl1_node = splitter.root.children[0]
        assert len(lvl1_node.children) > 0
        assert isinstance(lvl1_node.children[0], DocNode) is True
        assert lvl1_node.children[0].name == 'I. legislativo'

    def test_detect_content_subtitle(self):
        """
        Test that when a text is centered and immediately over a block of text
        that starts with 'Artículo ...' or the equivalent, it is considered as it's title.
        """
        # Reglamento del personal académico (Pag 9)
        data = pd.DataFrame([
            {"page": 2, "text": "Fundamento",
                "left": 0.204840878529807,
                "top": 0.407685512367491,
                "right": 0.292245629762438,
                "bottom": 0.418727915194346,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "objeto",
                "left": 0.309577170177798,
                "top": 0.407685512367491,
                "right": 0.352457791722695,
                "bottom": 0.421598939929329,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "Artículo",
                "left": 0.102046914686986,
                "top": 0.426568021201413,
                "right": 0.172269535335425,
                "bottom": 0.438825088339223,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "1.",
                "left": 0.178395338413268,
                "top": 0.43142667844523,
                "right": 0.189152846257284,
                "bottom": 0.438825088339223,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "re-",
                "left": 0.430150903929479,
                "top": 0.431537102473498,
                "right": 0.452562378604512,
                "bottom": 0.438825088339223,
            "line": 1, "column": 0, "col_position": 0, "group": 0},
            # Dummy words to stablish page writable area
            {"page": 2, "text": "dummy_left",
                "left": 0.102046914686986,
                "top": 0.207597173144876,
                "right": 0.172269535335425,
                "bottom": 0.438825088339223,
            "line": 2, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "dummy_right",
                "left": 0.815628268340057,
                "top": 0.905918727915194,
                "right": 0.852084267144778,
                "bottom": 0.917292402826855,
            "line": 3, "column": 0, "col_position": 0, "group": 2},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert len(splitter.root.children) > 0
        assert isinstance(splitter.root.children[0], DocNode) is True
        assert splitter.root.children[0].name == 'Artículo 1'
        assert splitter.root.children[0].title == 'Fundamento objeto'

    def test_detect_roman_numerals_in_column_as_simple_text(self):
        """
        Test that when a list of roman numerals is present as part of the content
        (kind of centered), it is not detected as a title of level 2.
        """
        # Reglamento del personal académico (Pag 18)
        data = pd.DataFrame([
            {"page": 2, "text": "berá",
                "left": 0.147766323024055,
                "top": 0.586020318021201,
                "right": 0.184670551322277,
                "bottom": 0.598277385159011,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "información:",
                "left": 0.376961004034065,
                "top": 0.586020318021201,
                "right": 0.489018377409234,
                "bottom": 0.598277385159011,
            "line": 0, "column": 0, "col_position": 0, "group": 0},
            {"page": 2, "text": "I.",
                "left": 0.176004781114597,
                "top": 0.626766784452297,
                "right": 0.18691169878978,
                "bottom": 0.638140459363958,
            "line": 1, "column": 0, "col_position": 0, "group": 1},
            {"page": 2, "text": "acadé-",
                "left": 0.441356641266995,
                "top": 0.625883392226148,
                "right": 0.496040639474077,
                "bottom": 0.638361307420495,
            "line": 1, "column": 0, "col_position": 0, "group": 1},
            # Dummy words to stablish page writable area
            {"page": 2, "text": "institucionales",
                "left": 0.147766323024055,
                "top": 0.107553003533569,
                "right": 0.273569400866577,
                "bottom": 0.120030918727915,
            "line": 2, "column": 0, "col_position": 0, "group": 2},
            {"page": 2, "text": "forma-",
                "left": 0.837442103690423,
                "top": 0.925022084805654,
                "right": 0.89571193784551,
                "bottom": 0.937279151943463,
            "line": 3, "column": 0, "col_position": 0, "group": 3},
        ])

        splitter = DataTreeSplitter(data)
        splitter.analyze()

        assert splitter.root.content.startswith('berá información:\nI. acadé-') is True

if __name__ == '__main__':
    unittest.main()
