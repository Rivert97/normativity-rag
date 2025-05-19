import re

METADATA_REGEX = {
    'permissive_titles' : {
        'titles': {
            2: r'^(t[iíÍ]tulo|[xiv]+\.) .*',
            3: r'^secci[oóÓ]n .*',
            4: r'^cap[iíÍ]tulo .*',
        },
        'contents': {
            5: r'^art[iíÍ]culo ([0-9]+|[a-záéíóú]+(ro|do|to|mo|vo|no)|[Úú]nico) ?(bis|ter|qu[aá]ter|quinquies)?\.',
        }
    },
    'not_permissive_titles' : {
        'titles': {
            2: r'^t[iíÍ]tulo [a-záéíóú]+(ro|do|to|mo|vo|no)$',
            3: r'^secci[oóÓ]n .*',
            4: r'^cap[iíÍ]tulo .*',
        },
        'contents': {
            5: r'^art[iíÍ]culo ([0-9]+|[a-záéíóú]+(ro|do|to|mo|vo|no)|[Úú]nico) ?(bis|ter|qu[aá]ter|quinquies)?\.',
        }
    },
}

class TitleDetector:

    def __init__(self, regex_type:str='permissive_titles'):
        self.metadata_regex = METADATA_REGEX.get(regex_type, METADATA_REGEX['permissive_titles'])

    def get_title_level(self, text:str) -> int:
        level = 1
        for lvl in self.metadata_regex['titles']:
            if re.match(self.metadata_regex['titles'][lvl], text.lower()):
                level = lvl
                break

        return level

    def get_number_of_titles(self):
        return len(self.metadata_regex['titles']) + 2

    def get_number_of_content_tiltes(self):
        return len(self.metadata_regex['contents'])

    def get_level_of_titles(self):
        return self.metadata_regex['titles'].keys()

    def get_level_of_contents(self):
        return self.metadata_regex['contents'].keys()

    def detect_content_header(self, text):
        level = -1
        name = ''
        content = ''

        for lvl in self.metadata_regex['contents'].keys():
            matches = re.search(self.metadata_regex['contents'][lvl], text.lower())
            if matches:
                l_match = len(matches.group(0))
                level = lvl
                name = text[:l_match].rstrip('.')
                content = text[l_match:].strip()
                break
        else:
            content = text

        return level, name, content