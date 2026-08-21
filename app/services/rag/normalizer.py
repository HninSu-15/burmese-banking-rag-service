import unicodedata
import re  # ← Add this import
class BurmeseTextNormalizer:
    """Utility class to sanitize and normalize Burmese text inputs and documents."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Applies Standard Unicode NFC Normalization to enforce encoding consistency."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Collapses duplicate spaces, newlines, and tabs into clean single spaces."""
        if not text:
            return ""
        # Replace newlines/tabs with space
        cleaned = re.sub(r'[\r\t\n]+', ' ', text)
        # Collapse multiple spaces into one
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def clean(cls, text: str) -> str:
        """Executes full normalization pipeline on raw text."""
        normalized = cls.normalize_unicode(text)
        cleaned = cls.clean_whitespace(normalized)
        return cleaned


# Independent Module Unit Test
if __name__ == "__main__":
    raw_sample = "ATM  ကတ်\nပျောက်သွားရင်   ဘာလုပ်ရမလဲ။"
    processed = BurmeseTextNormalizer.clean(raw_sample)
    print("Raw Text Input:   ", repr(raw_sample))
    print("Processed Output: ", repr(processed))