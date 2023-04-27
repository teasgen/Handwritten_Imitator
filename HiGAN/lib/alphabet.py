import unicodedata
import torch

# -\'.ü!"#%&()*+,/:;?
Alphabets = {
    # '!#&():;?*%'
    'all': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',  # n_class: 80
    'iam_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',  # n_class: 80
    'iam_line': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',  # n_class: 80
    'cvl_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?',  # n_class: 80
    'rimes_word': '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%\'-/Éàâçèéêëîïôùû'  # n_class: 81
}


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet_key, ignore_case=False):
        alphabet = Alphabets[alphabet_key]
        # print(alphabet)
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text, max_len=None):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if len(text) == 1:
            text = text[0]

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            return text

        length = []
        result = []
        results = []
        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
            results.append(result)
            result = []

        labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(text) for text in results], batch_first=True)
        lengths = torch.IntTensor(length)

        if max_len is not None and max_len > labels.size(-1):
            pad_labels = torch.zeros((labels.size(0), max_len)).long()
            pad_labels[:, :labels.size(-1)] = labels
            labels = pad_labels

        return labels, lengths
