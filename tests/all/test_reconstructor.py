"""Module to add unit test for DataReconstructor class."""
import unittest

import pandas as pd
import numpy as np

from simplerag.document_loaders.parsers import DataReconstructor

#pylint: disable=protected-access

class TestReconstructor(unittest.TestCase):
    """Class for unittesting DataReconstructor class.

    Important private methods are tested.
    """
    def test_is_same_line_uppercase(self):
        """Test that text in uppercase in the same line is correct."""
        reconstructor = DataReconstructor(pd.DataFrame())
        # Ley organica (Pag. 3): DICTAMEN DE
        result = reconstructor._DataReconstructor__is_same_line(
            line_top=0.147747349823322, line_bottom=0.159673144876325,
            word_top=0.148299469964664, word_height=0.0112632508833922
        )
        assert result is True

    def test_is_same_line_lowercase(self):
        """Test that text in lowercas in the same line is correct."""
        reconstructor = DataReconstructor(pd.DataFrame())
        # Ley organica (Pag. 3): I. En
        result = reconstructor._DataReconstructor__is_same_line(
            line_top=0.630742049469965, line_bottom=0.638140459363958,
            word_top=0.626766784452297, word_height=0.0111528268551237
        )
        assert result is True

    def test_is_same_line_upperindex(self):
        """Test that text with upperindices is considered in the same line."""
        reconstructor = DataReconstructor(pd.DataFrame())
        # Código de Ética de las personas... (Pag. 4): 2 Véase
        result = reconstructor._DataReconstructor__is_same_line(
            line_top=0.72482332155477, line_bottom=0.731448763250884,
            word_top=0.727583922261484, word_height=0.00938604240282686
        )
        assert result is True

    def test_is_not_same_line(self):
        """"Test that text in another line is correct."""
        reconstructor = DataReconstructor(pd.DataFrame())
        # Ley organica (Pag. 3): titucion...\n iniciativa...
        result = reconstructor._DataReconstructor__is_same_line(
            line_top=0.646863957597173, line_bottom=0.658348056537103,
            word_top=0.667292402826855, word_height=0.0107111307420495
        )
        assert result is False

    def test_assign_single_column_number(self):
        """Test that words that are close to each other are considered in the same column."""
        # Ley organica (Pag 10): potencial humano \n ciencia y
        data = pd.DataFrame([
            {"line": 0, "left": 987, "width": 535},
            {"line": 0, "left": 1580, "width": 488},
            {"line": 1, "left": 989, "width": 397},
            {"line": 1, "left": 1429, "width": 70},
        ])
        writable_boundaries = (989, 488, 5998, 8307)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_number()

        assert "column" in reconstructor.data
        assert reconstructor.data.loc[0, "column"] == reconstructor.data.loc[1, "column"]
        assert reconstructor.data.loc[2, "column"] == reconstructor.data.loc[3, "column"]

    def test_assign_two_column_number(self):
        """Test that words that are too far apart are considered in different columns."""
        # Ley organica (Pag 26): in-    XV. \n otras     o
        data = pd.DataFrame([
            {"line": 0, "left": 3161, "width": 159},
            {"line": 0, "left": 3702, "width": 224},
            {"line": 1, "left": 3042, "width": 278},
            {"line": 1, "left": 3969, "width": 64},
        ])
        writable_boundaries = (991, 488, 5996, 8490)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_number()

        assert "column" in reconstructor.data
        assert reconstructor.data.loc[0, "column"] != reconstructor.data.loc[1, "column"]
        assert reconstructor.data.loc[2, "column"] != reconstructor.data.loc[3, "column"]

    def test_assign_column_on_page_number_left(self):
        """Test that headers with page_number+title are treated as two columns."""
        # Ley organica (Pag 10): 10     Normatividad de
        data = pd.DataFrame([
            {"line": 0, "left": 989, "width": 115},
            {"line": 0, "left": 2102, "width": 835},
        ])
        writable_boundaries = (989, 488, 5998, 8307)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_number()

        assert "column" in reconstructor.data
        assert reconstructor.data.loc[0, "column"] != reconstructor.data.loc[1, "column"]

    def test_assign_column_on_page_number_right(self):
        """Test that headers with title+page_number are treated as single column."""
        # Ley organica (Pag 11): Ley Organica      11
        data = pd.DataFrame([
            {"line": 0, "left": 2813, "width": 196},
            {"line": 0, "left": 3060, "width": 515},
            {"line": 0, "left": 5595, "width": 65},
        ])
        writable_boundaries = (849, 494, 5703, 8308)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_number()

        assert "column" in reconstructor.data
        assert reconstructor.data.loc[0, "column"] == reconstructor.data.loc[1, "column"]

    def test_assign_column_position_on_centered_text(self):
        """Test that text that is centered is considered '0' aligned."""
        # Ley organica (Pag 3): DICTAMEN
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "left": 2662, "right": 3740},
        ])
        writable_boundaries = (698, 1338, 5705, 7948)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_position()

        assert "col_position" in reconstructor.data
        assert reconstructor.data.loc[0, "col_position"] == 0

    def test_assign_column_position_on_justified_text(self):
        """Test that text that is justified is considered '0' aligned."""
        # Ley organica (Pag 3): La ... estu-
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "left": 697, "right": 843},
            {"line": 0, "column": 0.0, "left": 5426, "right": 5703},
        ])
        writable_boundaries = (698, 1338, 5705, 7948)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_position()

        assert "col_position" in reconstructor.data
        assert reconstructor.data.loc[0, "col_position"] == 0

    def test_assign_column_position_on_left_aligned_text(self):
        """Test that text that is left aligned is considered '0' aligned."""
        # Ley organica (Pag 3): DICTAMEN ... GOBERNACIÓN
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "left": 698, "right": 1513},
            {"line": 0, "column": 0.0, "left": 3094, "right": 4208},
        ])
        writable_boundaries = (698, 1338, 5705, 7948)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_position()

        assert "col_position" in reconstructor.data
        assert reconstructor.data.loc[0, "col_position"] == 0

    def test_assign_column_position_on_right_aligned_text(self):
        """Test that text that is right aligned is considered '1' aligned."""
        # Ley organica (Pag 47): JUAN ... RAMIREZ
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "left": 3387, "right": 3773},
            {"line": 0, "column": 0.0, "left": 5011, "right": 5706},
        ])
        writable_boundaries = (697, 494, 5706, 3793)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_position()

        assert "col_position" in reconstructor.data
        assert reconstructor.data.loc[0, "col_position"] == 1

    def test_assign_column_position_on_right_column(self):
        """Test that text in right column is considered '1' aligned."""
        # Ley organica (Pag 23): Artículo 7.
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "left": 3358, "right": 3827},
            {"line": 0, "column": 0.0, "left": 3903, "right": 3978},
        ])
        writable_boundaries = (827, 494, 5674, 7983)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_column_position()

        assert "col_position" in reconstructor.data
        assert reconstructor.data.loc[0, "col_position"] == 1

    def test_assign_new_group_after_horizontal_line(self):
        """Test that a new group is created after a horizontal line is found in the document."""
        # Ley organica (Pag 23): pecíficos \n CAPITULO II
        data = pd.DataFrame([
            {"line": 0, "column": 1.0, "col_position": 1,
                "left": 0.503511131032422,
                "right": 0.583295980875542,
                "top": 0.187279151943463,
                "bottom": 0.203621908127208},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 0.4223815927088,
                "right": 0.512774540564769,
                "top": 0.247791519434629,
                "bottom": 0.259386042402827},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 0.520245032123114,
                "right": 0.533990736590468,
                "top": 0.248012367491166,
                "bottom": 0.25916519434629},
        ])
        lines = {
            "horizontal": np.array([[
                    [0.35454953, 0.21333922],
                    [0.60167339, 0.21333922]
                ]],
            )
        }
        writable_boundaries = (827, 494, 5674, 7983)

        reconstructor = DataReconstructor(data, writable_boundaries, lines)
        reconstructor._DataReconstructor__assign_group_number()

        assert "group" in reconstructor.data
        assert reconstructor.data.loc[0, "group"] != reconstructor.data.loc[1, "group"]

    def test_assign_new_group_when_col_number_change_and_one_pass_center(self):
        """
        Test that a new group is created when the number of column changes and at least
        one of them passes through the center.
        """
        # Ley organica (Pag 23): COMUNIDAD UNIVERSITARIA \n universitaria \t cial
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 2334,
                "right": 3097,
                "top": 2424,
                "bottom": 2529},
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 3146,
                "right": 4075,
                "top": 2426,
                "bottom": 2529},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 2309,
                "right": 3035,
                "top": 2793,
                "bottom": 2892},
            {"line": 1, "column": 1.0, "col_position": 1,
                "left": 3372,
                "right": 3570,
                "top": 2779,
                "bottom": 2890},
        ])
        writable_boundaries = (827, 494, 5674, 7983)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_group_number()

        assert "group" in reconstructor.data
        assert reconstructor.data.loc[0, "group"] == reconstructor.data.loc[1, "group"]
        assert reconstructor.data.loc[1, "group"] != reconstructor.data.loc[2, "group"]
        assert reconstructor.data.loc[2, "group"] == reconstructor.data.loc[3, "group"]

    def test_assign_new_group_when_change_between_centered_and_no(self):
        """
        Test that a new group is created when one line is centered and the next is not
        or the other way around.
        """
        # Ley organica (Pag 3): I. DEL PROCESO LEGISLATIVO \n ANTECEDENTES
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 2334,
                "right": 2413,
                "top": 4954,
                "bottom": 5057},
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 2469,
                "right": 2717,
                "top": 4954,
                "bottom": 5056},
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 2762,
                "right": 3274,
                "top": 4987,
                "bottom": 5057},
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 3323,
                "right": 4061,
                "top": 4985,
                "bottom": 5057},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 689,
                "right": 1608,
                "top": 5313,
                "bottom": 5418},
        ])
        writable_boundaries = (827, 494, 5674, 7983)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_group_number()

        assert "group" in reconstructor.data
        assert list(reconstructor.data[reconstructor.data['group'] == 0].index) == [0, 1, 2, 3]
        assert reconstructor.data.loc[3, "group"] != reconstructor.data.loc[4, "group"]

    def test_assign_same_group_when_change_centering_but_line_is_right_aligned(self):
        """
        Test that a the group mantained when prev line is centered but following line
        is right aligned. This means in an annotation.
        """
        # Ley organica (Pag 40): CAPITULO II \n ... 27-12-2016
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 3118,
                "right": 3723,
                "top": 3869,
                "bottom": 3974},
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 3774,
                "right": 3866,
                "top": 3871,
                "bottom": 3972},
            {"line": 1, "column": 0.0, "col_position": 1,
                "left": 5461,
                "right": 5994,
                "top": 4392,
                "bottom": 4487},
        ])
        writable_boundaries = (987, 488, 5995, 8488)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_group_number()

        assert "group" in reconstructor.data
        assert reconstructor.data.loc[0, "group"] == reconstructor.data.loc[1, "group"]
        assert reconstructor.data.loc[1, "group"] == reconstructor.data.loc[2, "group"]

    def test_assign_new_group_when_same_num_cols_but_diff_position(self):
        """
        Test that a the a new group is created when the number of columns is the
        same but the columns are not aligned with its corresponding column.
        """
        # Ley organica (Pag 20): 20    Normatividad ... Guanajuato \n DIP. ANTONIO    DIP. JOSÉ
        data = pd.DataFrame([
            {"line": 0, "column": 0.0, "col_position": 0,
                "left": 991,
                "right": 1123,
                "top": 484,
                "bottom": 620},
            {"line": 0, "column": 1.0, "col_position": 0,
                "left": 2102,
                "right": 2937,
                "top": 488,
                "bottom": 599},
            {"line": 0, "column": 1.0, "col_position": 0,
                "left": 4189,
                "right": 4865,
                "top": 494,
                "bottom": 638},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 1636,
                "right": 1896,
                "top": 1343,
                "bottom": 1446},
            {"line": 1, "column": 0.0, "col_position": 0,
                "left": 1940,
                "right": 2672,
                "top": 1338,
                "bottom": 1446},
            {"line": 1, "column": 1.0, "col_position": 1,
                "left": 3988,
                "right": 4249,
                "top": 1343,
                "bottom": 1446},
            {"line": 1, "column": 1.0, "col_position": 1,
                "left": 4291,
                "right": 4629,
                "top": 1313,
                "bottom": 1475},
        ])
        writable_boundaries = (991, 484, 5632, 2710)

        reconstructor = DataReconstructor(data, writable_boundaries)
        reconstructor._DataReconstructor__assign_group_number()

        assert "group" in reconstructor.data
        assert list(reconstructor.data[reconstructor.data['group'] == 0].index) == [0, 1, 2]
        assert list(reconstructor.data[reconstructor.data['group'] == 1].index) == [3, 4, 5, 6]
        assert reconstructor.data.loc[2, "group"] != reconstructor.data.loc[3, "group"]

if __name__ == '__main__':
    unittest.main()
