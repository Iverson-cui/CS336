from heapq import merge
from math import e
from operator import index
import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
import numpy as np
import multiprocessing
import cProfile
import pickle
from collections import namedtuple


NUM_PROCESSES = 8

# this namedtuple is used in merge functions
# bytes means the current bytes, and index means its index in list
# usage: bytes1=merge_bytes(b'th',3), _bytes1=bytes1.bytes _index1=bytes1.index
merge_bytes = namedtuple("merge_bytes", ["bytes", "index"])


def find_chunk_boundaries(
    f: str,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    return a list of int, with each element the index of the boundary for a given file
    """
    assert isinstance(
        split_special_tokens, list
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    with open(f, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                positions = [
                    pos
                    for token in split_special_tokens
                    if (pos := mini_chunk.find(token)) != -1
                ]
                if positions:
                    earliest_pos = min(positions)
                    chunk_boundaries[bi] = initial_position + earliest_pos
                    break

                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    """
    Worker function to process a single chunk of the file.
    Args: tuple containing (file_path, start, end, pattern)
    Returns: dictionary mapping pre-tokenized strings to their frequencies
    """
    file_path, start, end, pattern = args
    chunk_frequency_dict = {}

    with open(file_path, "rb") as f:
        f.seek(start)
        # chunk is of type string
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # @ before running pre-tokenization we first need to split based on special tokens
        special_tokens = ["<|endoftext|>"]
        delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
        # Split the chunk into document segments based on special tokens
        document_segments = re.split(delimiter_pattern, chunk)
        # Pre-tokenize each segment separately
        # @ Run pre-tokenization on your chunk and store the counts for each pre-token
        for segment in document_segments:
            if segment.strip():  # Skip empty segments
                # chunk_after_preto is a list of strings
                # whose elements are word string divided by pre-tokenization
                # finditer is an iterator of strings.
                chunk_after_preto = re.finditer(pattern, segment)
                for i, match in enumerate(chunk_after_preto):
                    chunk_frequency_dict[
                        match.group().encode("utf-8", errors="ignore")
                    ] = (
                        chunk_frequency_dict.get(
                            match.group().encode("utf-8", errors="ignore"), 0
                        )
                        + 1
                    )

    return chunk_frequency_dict


def generate_pre_frequency_dict(file: str, pattern, boundaries: list[int]):
    """
    given the boundaries and file name, return a dictionary mapping pre-tokenized strings to their frequencies
    the element in the dictionary: first element is the bytes, second element is the frequency int.
    """
    # Prepare arguments for each chunk
    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    chunk_args = [(file, start, end, pattern) for start, end in chunk_pairs]

    # Use multiprocessing to process chunks in parallel
    with multiprocessing.Pool() as pool:
        chunk_results = pool.map(process_chunk, chunk_args)

    # Merge all chunk results into a single dictionary
    pre_frequency_dict = {}
    for chunk_dict in chunk_results:
        for token_bytes, frequency in chunk_dict.items():
            pre_frequency_dict[token_bytes] = (
                pre_frequency_dict.get(token_bytes, 0) + frequency
            )

    return pre_frequency_dict


def generate_byte_pair_frequency_tensor(pre_frequency_dict: dict):
    """
    This function takes in a dictionary that keeps the pre-tokenization result and return a sorted
    numpy array which keeps the frequency of each byte pair.
    In this function, first a dictionary of byte pair frequency is generated, then a numpy array is
    generated based on that intermediate dictionary.
    Input:
        pre_frequency_dict: a dictionary, whose keys are tuples of int and whose values are int
    Output:
        A 3D numpy tensor of shape (num_pairs, 3) where each row is [byte1, byte2, frequency]
        This tensor has elements of type np.int32.
    """
    pair_frequency_dict = {}
    for token_bytes, frequency in pre_frequency_dict.items():
        # Skip if token has only one byte
        if len(token_bytes) <= 1:
            continue
        # Get all byte pairs using zip
        # byte_pairs is a list of 2 element tuples, each element is of type int
        byte_pairs = list(zip(token_bytes[:-1], token_bytes[1:]))

        # Update frequency for each byte pair
        # byte_pair is of type tuple(bytes,bytes)
        for byte_pair in byte_pairs:
            # pair_frequency_dict's key is of type tuple(int,int)
            # list(bytes) will transform bytes to int
            pair_frequency_dict[byte_pair] = (
                pair_frequency_dict.get(byte_pair, 0) + frequency
            )

    # Sort pair_frequency_dict by values (frequencies) in descending order
    # sorted_pairs is a list of tuples
    # it's like [ ( ( 32,116 ),641902 ), ( ( 97,98 ),123456 ), ... ]
    sorted_pairs = sorted(pair_frequency_dict.items(), key=lambda x: x[1], reverse=True)
    # Print some results
    # print("Top 10 byte pairs:")
    # for i, (byte_pair, freq) in enumerate(sorted_pairs[:10]):
    #     # Convert bytes to characters for readability (if possible)
    #     try:
    #         char1 = (
    #             chr(byte_pair[0])
    #             if 32 <= byte_pair[0] <= 126
    #             else f"\\x{byte_pair[0]:02x}"
    #         )
    #         char2 = (
    #             chr(byte_pair[1])
    #             if 32 <= byte_pair[1] <= 126
    #             else f"\\x{byte_pair[1]:02x}"
    #         )
    #         print(
    #             f"  {i+1}. ({byte_pair[0]}, {byte_pair[1]}) -> '{char1}{char2}': {freq}"
    #         )
    #     except:
    #         print(f"  {i+1}. {byte_pair}: {freq}")

    # Convert sorted_pairs to numpy tensor
    # Extract the data into separate arrays
    byte_pairs = np.array([pair for pair, freq in sorted_pairs], dtype=np.int32)
    frequencies = np.array([freq for pair, freq in sorted_pairs], dtype=np.int32)

    # Create the 3D tensor by stacking
    # Shape will be (num_pairs, 3) where each row is [byte1, byte2, frequency]
    # but now byte1 and byte2 are all of type np.int32
    frequency_tensor = np.column_stack([byte_pairs, frequencies])

    return frequency_tensor


def generate_merged_pre_frequency_dict(
    pre_frequency_dict: dict, merge_byte_pair: tuple, merge_index: int
):
    """
    This function generates the new pre-tokenization dict based on the old one and the pair we
    want to merge.
    Inputs:
        pre_frequency_dict: The original pre-tokenization frequency dictionary, keys are bytes and values are int
        merge_byte_pair: The byte pair to merge (as a tuple of two ints).
        merge_index: The index of the merge operation used to decide new token.

    Output:
        new_pre_frequency_dict: The new pre-tokenization frequency dictionary with key tuples and value int
        The tuple contains multiple ints which is larger than 256.
    """
    new_pre_frequency_dict = {}
    for token_bytes, frequency in pre_frequency_dict.items():
        # list() convert bytes to list of ints
        token_list = list(token_bytes)
        # Find and merge byte pairs and modify it
        i = 0
        while i < len(token_list) - 1:
            if (token_list[i], token_list[i + 1]) == (
                int(merge_byte_pair[0]),
                int(merge_byte_pair[1]),
            ):
                # Replace the pair with new token (256 + merge_index)
                token_list[i] = 256 + merge_index
                # Remove the second byte of the pair
                token_list.pop(i + 1)
            else:
                i += 1
        # Original key: bytes are transformed to tuples of int, which may be larger than 256
        new_token_key = tuple(token_list)
        new_pre_frequency_dict[new_token_key] = frequency

    return new_pre_frequency_dict


def merge_n_times(
    initial_pre_frequency_dict: dict, n: int, len_special_tokens: int, vocab: dict
):
    """
    This function merges the most frequent byte pair n times.
    Inputs:
        initial_pre_frequency_dict: The initial pre-tokenization frequency dictionary.
        n: The number of times to merge the most frequent byte pair.
        len_special_tokens: The length of the special tokens list.
        vocab: The current vocabulary dictionary mapping token IDs to byte sequences.

    Output:
        new_pre_frequency_dict: The new pre-tokenization frequency dictionary after merging.
    """
    # @ keep the frequency of current byte pairs by another dictionary
    # initial_pre_frequency_dict has keys bytes and values int
    frequency_tensor = generate_byte_pair_frequency_tensor(initial_pre_frequency_dict)
    # merge_byte_pair is a tuple containing 2 int standing for the byte pair to merge
    # Although frequency_tensor[0][0] is of type np.int32, there is no bug
    # That's because now no merge is done, so all int is smaller than 256 thus can -> bytes
    # merge_byte_pair is a tuple of 2 bytes
    merge_byte_pair = (bytes([frequency_tensor[0][0]]), bytes([frequency_tensor[0][1]]))
    # merges is a list of tuples which contain 2 bytes
    merges = []
    merges.append(merge_byte_pair)

    # Add the merged pair to vocabulary
    new_token_id = 256 + len_special_tokens
    # vocab's elements are of type bytes
    vocab[new_token_id] = merge_byte_pair[0] + merge_byte_pair[1]

    # @ first, merge 1 byte pair and generate new byte pair frequency after merge
    # 2 things need to be done if we want to merge the byte pair:
    # 1. Based on the pair to merge, generate new pre-tokenization dict
    # 2. Based on the new pre dict, generate the byte pair frequency dict
    new_pre_frequency_dict = generate_merged_pre_frequency_dict(
        initial_pre_frequency_dict,
        (frequency_tensor[0][0], frequency_tensor[0][1]),
        len_special_tokens,
    )
    # @ rest merge in iteration
    for _ in range(n - 1):
        new_frequency_tensor = generate_byte_pair_frequency_tensor(
            new_pre_frequency_dict
        )
        # new_merge_byte_pair is a tuple containing 2 np.int32 elements
        new_merge_byte_pair_ints = (
            new_frequency_tensor[0][0],
            new_frequency_tensor[0][1],
        )
        # print(f"New merge byte pair (ints): {new_merge_byte_pair_ints}")
        # but when storing it in merges and vocab, we want bytes objects
        # so the bytes object may contain multiple characters
        new_merge_byte_pair_bytes = (
            vocab[new_frequency_tensor[0][0]],
            vocab[new_frequency_tensor[0][1]],
        )
        # print(f"New merge byte pair (bytes): {new_merge_byte_pair_bytes}")
        merges.append(new_merge_byte_pair_bytes)

        new_token_id = 256 + len_special_tokens + _ + 1
        vocab[new_token_id] = (
            new_merge_byte_pair_bytes[0] + new_merge_byte_pair_bytes[1]
        )

        # We need to use new pre_frequency_dict containing 1st merge as input
        new_new_pre_frequency_dict = generate_merged_pre_frequency_dict(
            new_pre_frequency_dict, new_merge_byte_pair_ints, len_special_tokens + _ + 1
        )

        new_pre_frequency_dict = new_new_pre_frequency_dict
        # new_new_frequency_tensor = generate_byte_pair_frequency_tensor(
        #     new_new_pre_frequency_dict
        # )
    return merges


def train(input_path: str, vocab_size: int, special_tokens: list[bytes]):
    f = input_path
    merge_time = vocab_size - 256 - len(special_tokens)

    # Initialize vocab with primitive 256 bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}

    # Add special tokens starting from token ID 256
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token

    # This is the pattern used to divide words, i.e. pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # boundaries is a list containing the location of boundaries in the txt file
    boundaries = find_chunk_boundaries(f, NUM_PROCESSES, special_tokens)
    # @ pre-tokenization to the text
    # @ pre_frequency_dict with no merge operation done
    pre_frequency_dict = generate_pre_frequency_dict(f, PAT, boundaries)
    # @ final pre_frequency_dict after merging n times
    merges = merge_n_times(pre_frequency_dict, merge_time, len(special_tokens), vocab)
    return vocab, merges


def main():
    """
    For the use of multiprocessing we have to name the whole function main
    """
    final_vocab, merges = train("../data/valid.txt", 300, [b"<|endoftext|>"])
    # print("Final Vocabulary:")
    # print(f"Vocabulary size: {len(final_vocab)}")
    # print("\nSpecial and merged tokens:")
    # for token_id, token_bytes in final_vocab.items():
    #     if token_id >= 256:  # Only show special tokens and merged tokens
    #         try:
    #             # Try to decode as UTF-8 for readability
    #             token_str = token_bytes.decode("utf-8", errors="replace")
    #             print(f"  {token_id}: {token_bytes} -> '{token_str}'")
    #         except:
    #             print(f"  {token_id}: {token_bytes}")

    # print(f"\nMerges performed ({len(merges)} total):")
    # for i, (byte1, byte2) in enumerate(merges):
    #     try:
    #         # Try to show readable characters
    #         char1 = byte1.decode("utf-8", errors="replace")
    #         char2 = byte2.decode("utf-8", errors="replace")
    #         print(f"  {i+1}. {byte1} + {byte2} -> '{char1}' + '{char2}'")
    #     except:
    #         print(f"  {i+1}. {byte1} + {byte2}")

    # generate tokenizer object
    tokenizer = Tokenizer(final_vocab, merges, special_tokens=["<|endoftext|>"])
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    print(encoded_ids)
    decoded_string = tokenizer.decode(encoded_ids)
    print(decoded_string)
    assert test_string == decoded_string


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        INPUTS:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # Create reverse lookup: bytes -> token_id
        self.bytes_to_id = {value: key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        This is Tokenizer's additional constructor that takes argument from file rather
        than directly input it.
        This is a class method.
        """

        import pickle

        # Load vocabulary using pickle for binary data handling
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        # Load merges using pickle
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Given a str of text that we want to encode, this function returns the corresponding
        list of token IDs.
        Special tokens must be handled very well.
        """
        print(f"Encoding text: {text}")
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # @ before running pre-tokenization we first need to split based on special tokens
        # Create capturing group pattern to preserve special tokens
        # This is different from above for we want to preserve special tokens
        delimiter_pattern = (
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        )
        # Split the chunk into document segments based on special tokens
        # Pre-tokenize each segment separately
        text_segments = re.split(delimiter_pattern, text)
        tokenid_list = []
        for segment in text_segments:
            if not segment:  # Skip empty segments
                continue
            # if this segment is special tokens
            if segment in self.special_tokens:
                # Handle special token - find its token ID
                special_token_bytes = segment.encode("utf-8")
                if special_token_bytes in self.bytes_to_id:
                    token_id = self.bytes_to_id[special_token_bytes]
                    # update tokenid_list based on this special token
                    tokenid_list.append(token_id)
                continue
            # @ for each segment, pretokenize first
            # here, the segment is text instead of special tokens
            # if segment.strip():  # Skip empty segments
            # word is a generator containing pretokenization result
            word = re.finditer(PAT, segment)
            # for each pre-tokenization word
            for match in word:
                # match is the word string we are looking at
                assert isinstance(match.group(), str)
                word_text = match.group()
                print(f"This word text is {word_text}")

                # transform every word into a list of int (byte values)
                byte_list = [
                    self.bytes_to_id[bytes([b])]
                    for b in word_text.encode("utf-8", errors="ignore")
                ]
                # byte_list = list(word_text.encode("utf-8", errors="ignore"))
                print(f"Byte list: {byte_list}")
                # print(f"Current word: {word_text}")
                # print(f"Byte list: {byte_list}")
                # print("Next we enter into BPE merging stage")

                # Apply BPE merges to this word
                self._apply_bpe_merges(byte_list)
                # update tokenid_list to contain byte_list
                for i in byte_list:
                    tokenid_list.append(i)
        return tokenid_list

    def _apply_bpe_merges(self, byte_list):
        """
        Apply BPE merges to a list of bytes
        This is a sub function used in encode function
        """
        # total_byte_list is the temporary byte list
        if len(byte_list) == 0:
            return

        if len(byte_list) == 1:
            # Single byte
            single_byte = self.vocab[byte_list[0]]
            assert single_byte in self.bytes_to_id
            byte_list = [self.bytes_to_id[single_byte]]
            return

        # We modify the list in place
        # if the list is empty, we come to the end
        # first, Initialize byte1 and byte2 with first two bytes
        byte1 = merge_bytes(self.vocab[byte_list[0]], 0)
        print(f"Initial byte1: {byte1.bytes}")
        byte2 = merge_bytes(self.vocab[byte_list[1]], 1)
        print(f"Initial byte2: {byte2.bytes}")
        print("Initial two bytes are given, now we go into merge_end stage")
        # given two initial bytes, this function can merge until the end
        self._merge_two_bytes_till_the_end(byte1, byte2, byte_list)

    def _merge_two_bytes_till_the_end(
        self,
        byte1: merge_bytes,
        byte2: merge_bytes,
        byte_list: list,
    ):
        """
        Merge the whole byte list till the end.
        Inputs:
            byte1: a merge_bytes, with byte1.bytes and byte1.index
            byte2: a merge_bytes, with byte2.bytes and byte2.index
            byte_list: the original list of bytes, which is intact during the process

        Output:
            merge time in this round

        This function modifies byte_list in place, so no need to return anything.
        """

        print(f"Current byte list: {byte_list}")
        print(f"Attempting merge: {byte1.bytes}, {byte2.bytes}")

        # if two bytes can be merged together
        if (byte1.bytes, byte2.bytes) in self.merges:
            print(f"Bytes can be merged: {byte1.bytes}, {byte2.bytes}")
            merged_bytes = byte1.bytes + byte2.bytes
            assert merged_bytes in self.bytes_to_id

            # tokenid_list.append(self.bytes_to_id[merged_bytes])
            # if byte1 and byte2 are the last 2 bytes
            if len(byte_list) == 2:
                # only 2 bytes in the list, and they can merge
                # so only 1 element exists in byte_list
                print("This word only has 2 bytes, so merge is done now.")
                byte_list = [self.bytes_to_id[merged_bytes]]
                return
            # if not
            # update byte_list by:
            #   replace byte1 index with this new tokenid
            #   remove byte2 index byte
            tokenid = self.bytes_to_id[merged_bytes]
            print(f"Merging {byte1.bytes} + {byte2.bytes} -> token ID {tokenid}")
            print(
                f"As can be seen in vocab, {self.vocab[tokenid]} is the merged bytes in vocab."
            )
            byte_list[byte1.index] = tokenid
            byte_list.pop(byte2.index)
            print(f"Updated byte list after merge: {byte_list}")
            # update byte1 and byte2 for the next iteration
            byte1 = merge_bytes(merged_bytes, byte1.index)
            # print(f"Merge! Merged bytes: {byte1}")
            # first, perform backward merge if byte1 is not the first
            if not byte1.index == 0:
                print(
                    f"byte1 {byte1.bytes} is not the first so we need to do backward merge"
                )
                prev_bytes = merge_bytes(
                    self.vocab[byte_list[byte1.index - 1]], byte1.index - 1
                )
                print(
                    f"Previous bytes: {prev_bytes.bytes} with index {prev_bytes.index}"
                )
                while (prev_bytes.bytes, byte1.bytes) in self.merges:
                    print(
                        f"Backward merging can be done: {prev_bytes.bytes}, {byte1.bytes}"
                    )
                    # merged_bytes becomes bigger
                    merged_bytes = prev_bytes.bytes + byte1.bytes
                    assert merged_bytes in self.bytes_to_id
                    tokenid = self.bytes_to_id[merged_bytes]
                    print(f"Backward merge token ID: {tokenid}")
                    print(
                        f"corresponding merging bytes in my vocab: {self.vocab[tokenid]}"
                    )
                    # perform merge operation
                    byte_list[prev_bytes.index] = tokenid
                    byte_list.pop(byte1.index)
                    print(f"Byte list after backward merge: {byte_list}")
                    # update byte1 and prev_bytes
                    byte1 = merge_bytes(merged_bytes, prev_bytes.index)
                    # if we hit the start of the list
                    if prev_bytes.index == 0:
                        break
                    prev_bytes = merge_bytes(
                        self.vocab[byte_list[prev_bytes.index - 1]],
                        prev_bytes.index - 1,
                    )

                print("Backward is done. Now forward merge")
                if prev_bytes.index >= len(byte_list) - 2:
                    print("we have come to the end, merge is over!!")
                    return
                next_byte = merge_bytes(
                    self.vocab[byte_list[prev_bytes.index + 2]],
                    prev_bytes.index + 2,
                )
                prev_bytes = merge_bytes(
                    self.vocab[byte_list[prev_bytes.index + 1]],
                    prev_bytes.index + 1,
                )
                self._merge_two_bytes_till_the_end(prev_bytes, next_byte, byte_list)
                return
            # if byte1 is the first
            # do forward merge directly

            print(f"{byte1.bytes} is at first, no forward merge is possible")
            byte2 = merge_bytes(self.vocab[byte_list[byte2.index]], byte2.index)
            assert byte1.index + 1 == byte2.index
            # print(f"Next byte: {byte2}")
            self._merge_two_bytes_till_the_end(byte1, byte2, byte_list)
            return
        # if can not be merged
        print(f"Bytes cannot be merged: {byte1.bytes}, {byte2.bytes}")
        assert byte1.bytes in self.bytes_to_id
        if byte2.index == len(byte_list) - 1:
            print("we have come to the end, merge is over!!")
            return
        byte1 = byte2
        # print(f"No merge! now Current byte: {byte1}")
        byte2 = merge_bytes(self.vocab[byte_list[byte1.index + 1]], byte1.index + 1)
        # print(f"Next byte: {byte2}")
        self._merge_two_bytes_till_the_end(byte1, byte2, byte_list)
        return

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        This function, given the token id list of a paragraph of text, return
        the paragraph text string based on the vocabulary.
        Input:
            ids: a list of int whose element is the token ID.
        Output:
            A string representing the decoded text.
        """
        current_bytes = b""
        for tokenid in ids:
            assert tokenid in self.vocab
            corresponding_byte = self.vocab[tokenid]
            current_bytes += corresponding_byte
        return current_bytes.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    # cProfile.run("main()")
    # main()

    test_bytes = b"Hello, how are you The was?"
    test_string = "Hello, how are you The was?"
    final_vocab, merges = train("../data/valid.txt", 1000, [b"<|endoftext|>"])
    test_tokenizer = Tokenizer(final_vocab, merges, special_tokens=["<|endoftext|>"])
    encoded_ids = test_tokenizer.encode(test_string)
    print(encoded_ids)
    decoded_string = test_tokenizer.decode(encoded_ids)
    print(decoded_string)
    vocab = test_tokenizer.vocab
    merges = test_tokenizer.merges

    # encode_result = test_tokenizer.encode("Hello, how are you?")
