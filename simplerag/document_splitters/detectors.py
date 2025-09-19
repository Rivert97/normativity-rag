"""Module to define different types of section detectors in text files."""

import re

DEFAULT_TITLES_REGEX = {
    'titles': {
        1: r'^(?![a-z]\))(?!art[iíÍ]culo [0-9]+\.).*[^;:]$',
        2: str(r'^([iíÍ]ndice$|art[iíÍ]culos? transitorios?( de reforma)?)|'
                r'((\(.*\) )?(t[iíÍ]tulo|[xiv]+\.|[0-9]+\.) .*[^;:])$'),
        3: r'^([0-9]+\.[0-9]|secci[oóÓ]n) .*$',
        4: r'^(\(.*\) )?cap[iíÍ]tulo .*$',
    },
    'contents': {
        5: str(
            r'art[iíÍ]culo ([0-9]+|[a-záéíóú]+(ro|do|to|mo|vo|no)|[Úú]nico)[ \-]?'
            r'(bis|ter|qu[aá]ter|[a-záéíóú]+ies)?\.'),
    }
}

class TitleDetector:
    """Class to detect titles of sections and subsections using regular expressions."""

    def __init__(self, titles_regex:dict[str|int,str]|None=None):
        if titles_regex is None:
            self.metadata_regex = DEFAULT_TITLES_REGEX
        else:
            self.metadata_regex = titles_regex

    def get_title_level(self, text:str) -> int:
        """Get the level of the title, the deeper the title the higher the level."""
        level = 0
        for lvl in sorted(self.metadata_regex['titles'].keys(), reverse=True):
            if re.match(self.metadata_regex['titles'][lvl], text.lower()):
                level = lvl
                break

        return level

    def get_number_of_titles(self):
        """Get the number of different types of titles the dector can find."""
        return len(self.metadata_regex['titles']) + 2

    def get_number_of_content_titles(self):
        """Get the number of different types of sections in the contents the dector can find."""
        return len(self.metadata_regex['contents'])

    def get_level_of_titles(self):
        """Get the list of title levels the dector can find."""
        return self.metadata_regex['titles'].keys()

    def get_level_of_contents(self):
        """Get the list of section levels the detector can find."""
        return self.metadata_regex['contents'].keys()

    def detect_content_header(self, text):
        """Get the level of the section according to the text.

        Return -1 when the string is not a section header.
        """
        level = -1
        name = ''
        content = ''

        for lvl in sorted(self.metadata_regex['contents'].keys(), reverse=True):
            matches = re.search(r'^' + self.metadata_regex['contents'][lvl], text.lower())
            if matches:
                l_match = len(matches.group(0))
                level = lvl
                name = text[:l_match].rstrip('.')
                content = text[l_match:].strip()
                break
        else:
            content = text

        return level, name, content

    def detect_content_header_with_subtitle(self, text, max_subtitle_lines):
        """Get the level of the section according to the text when the text has a subtitle.

        Return -1 when the string is not a section header.
        """
        subtitle = ''
        content = ''

        for lvl in sorted(self.metadata_regex['contents'].keys(), reverse=True):
            regex = r'^([^\n]*\n{1,' + str(max_subtitle_lines) + r'})(' + self.metadata_regex['contents'][lvl] + r'.+)'
            matches = re.search(regex, text.lower())
            if matches:
                l_match_0 = len(matches.group(1))
                subtitle = text[:l_match_0].strip()
                content = text[l_match_0:]
                break
        else:
            content = text

        return subtitle, content
