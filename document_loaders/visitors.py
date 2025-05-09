class PageTextVisitor(object):
    """Visitor for pypdf.PdfReader.extract_text() function.

    Extracts relevant information of the document while reading it.
    This information can be later retrieved to perform certain operations.
    """
    def __init__(self):
        self.out_of_bounds_text = {}
        self.boundaries = (0.0, 0.0, 482.0, 652.0)

    def set_boundaries(self, left: float, top: float, right: float, bottom: float):
        """Set the boundaries of detection of text. Text outside the boundaries
        can be accessed through self.get_out_of_boundaries_text().

        Everything outside the square [(left, top) (right, bottom)] will be
        considered out of boundary.
        """
        self.boundaries = (top, left, bottom, right)

    def visitor_text(self, text, cm, tm, font_dict, font_size):
        x = tm[4]
        y = tm[5]
        if not (self.boundaries[1] <= x <= self.boundaries[3] and self.boundaries[0] <= y <= self.boundaries[2]):
            self.out_of_bounds_text.setdefault(y, []).append(text)

    def get_out_of_bounds_text(self):
        """Return an array of lines of text detected as out of boundaries by the visitor."""
        return [''.join(parts) for parts in self.out_of_bounds_text.values()]
