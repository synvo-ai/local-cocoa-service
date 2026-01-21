from rapidocr_onnxruntime import RapidOCR
import time


class IMG2WORDS():
    def __init__(self):
        self.engine = RapidOCR()

    def ocr_words_to_fitz_words(self, ocr_results):
        """
        Convert OCR word-level results to PyMuPDF 'words' format.

        Input:
            ocr_results: list of (text, score, quad)
                quad = [[x,y], [x,y], [x,y], [x,y]]

        Output:
            list of tuples:
                (x0, y0, x1, y1, text, block_no, line_no, word_no)
        """
        words = []

        for word_no, (text, score, quad) in enumerate(ocr_results):
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]

            x0 = float(min(xs))
            y0 = float(min(ys))
            x1 = float(max(xs))
            y1 = float(max(ys))

            words.append((
                x0, y0, x1, y1,
                text,          # word text (not single char)
                0,             # block_no placeholder
                0,             # line_no placeholder
                word_no,       # word_no by enumerate
            ))

        return words

    def run(self, img_dir):
        try:
            # Use word-level boxes instead of single-char boxes
            # This produces proper words like "Since" instead of "S" "i" "n" "c" "e"
            result = self.engine(img_dir, return_word_box=True)
            if result and result.word_results:
                words = self.ocr_words_to_fitz_words(result.word_results)
                return words
            return []
        except Exception as e:
            print(e)
            return []
