"""Module to define pypdf visitors."""

class PageTextVisitor:
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
        self.boundaries = (left, top, right, bottom)

    def visitor_text(self, *args):
        """Store in an array all the texts that are out of bounds.

        :param text: Retrieved text
        :type text: str
        :param cm: Current transformation matrix
        :type cm: list
        :param tm: Text transformation matrix
        :type tm: list
        :param font_dict: Specification of font types
        :type font_dict: dict
        :param font_size: Specification of font sizes
        :type font_size: dict
        """
        if args[0].strip() == '':
            return

        x = args[2][4]
        y = args[2][5]
        if not (self.boundaries[0] <= x <= self.boundaries[2] and
                self.boundaries[3] <= y <= self.boundaries[1]):
            self.out_of_bounds_text.setdefault(y, []).append(args[0].strip())

    def get_out_of_bounds_text(self):
        """Return an array of lines of text detected as out of boundaries by the visitor."""
        return [''.join(parts) for parts in self.out_of_bounds_text.values()]
