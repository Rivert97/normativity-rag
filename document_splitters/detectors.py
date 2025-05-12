import re

METADATA_REGEX = {
    'titles': {
        2: r'^(t[iíÍ]tulo|[xiv]+\.) .*',
        3: r'^secci[oóÓ]n .*',
        4: r'^cap[iíÍ]tulo .*',
    },
    'contents': {
        5: r'^art[iíÍ]culo ([0-9]+|[a-záéíóú]+(ro|do|ro|to|mo|vo|no)|[Úú]nico) ?(bis|ter|qu[aá]ter|quinquies)?\.',
    }
}

class TitleDetector:

    def get_title_level(self, text:str) -> int:
        level = 1
        for lvl in METADATA_REGEX['titles']:
            if re.match(METADATA_REGEX['titles'][lvl], text.lower()):
                level = lvl
                break

        return level

    def get_number_of_titles(self):
        return len(METADATA_REGEX['titles']) + 2

    def get_number_of_content_tiltes(self):
        return len(METADATA_REGEX['contents'])

    def get_level_of_titles(self):
        return METADATA_REGEX['titles'].keys()

    def get_level_of_contents(self):
        return METADATA_REGEX['contents'].keys()

    def detect_content_header(self, text):
        level = -1
        name = ''
        content = ''

        for lvl in METADATA_REGEX['contents'].keys():
            matches = re.search(METADATA_REGEX['contents'][lvl], text.lower())
            if matches:
                l_match = len(matches.group(0))
                level = lvl
                name = text[:l_match].rstrip('.')
                content = text[l_match:].strip()
                break
        else:
            content = text

        return level, name, content