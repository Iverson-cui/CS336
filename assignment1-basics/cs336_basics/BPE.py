import os
from typing import BinaryIO
import regex as re
import numpy as np
from tqdm import tqdm


def find_chunk_boundaries(
    f: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    return a list of int, with each element the index of the boundary for a given file
    """
    assert isinstance(
        split_special_token, bytes
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
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def generate_pre_frequency_dict(file: str, pattern, boundaries: list[int]):
    """
    given the boundaries and file name, return a dictionary mapping pre-tokenized strings to their frequencies
    the element in the dictionary: first element is the bytes, second element is the frequency int.
    """
    pre_frequency_dict = {}
    with open(file, "rb") as f:

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        # Add progress bar with total number of chunks
        chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        for start, end in tqdm(chunk_pairs, desc="Processing chunks", unit="chunk"):
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
                        # print(f"Match {i + 1}:")
                        # match.group() is of type string
                        # print(f"  Text: '{match.group()}'")
                        # string.encode() is of type bytes
                        # print(match.group().encode("utf-8", errors="ignore"))
                        pre_frequency_dict[
                            match.group().encode("utf-8", errors="ignore")
                        ] = (
                            pre_frequency_dict.get(
                                match.group().encode("utf-8", errors="ignore"), 0
                            )
                            + 1
                        )
        # print(
        #     f"  Start position: {match.start()}"
        # )  # Where match begins in original text
        # print(f"  End position: {match.end()}")  # Where match ends in original text
        # print(f"  Span: {match.span()}")  # (start, end) as a tuple
        # print()
    return pre_frequency_dict


def generate_byte_pair_frequency_tensor(pre_frequency_dict: dict):
    """
    This function takes in a dictionary that keeps the pre-tokenization result and return a sorted
    numpy array which keeps the frequency of each byte pair.
    In this function, first a dictionary of byte pair frequency is generated, then a numpy array is
    generated based on that intermediate dictionary.
    Input:
        pre_frequency_dict: a dictionary, whose keys are byte and whose values are int
    Output:
        A 3D numpy tensor of shape (num_pairs, 3) where each row is [byte1, byte2, frequency]
    """
    pair_frequency_dict = {}
    for token_bytes, frequency in pre_frequency_dict.items():
        # Skip if token has only one byte
        if len(token_bytes) <= 1:
            continue
        # Get all byte pairs using zip
        # byte_pairs is a list of 2 element tuples, each element is of type bytes
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
    print("Top 10 byte pairs:")
    for i, (byte_pair, freq) in enumerate(sorted_pairs[:10]):
        # Convert bytes to characters for readability (if possible)
        try:
            char1 = (
                chr(byte_pair[0])
                if 32 <= byte_pair[0] <= 126
                else f"\\x{byte_pair[0]:02x}"
            )
            char2 = (
                chr(byte_pair[1])
                if 32 <= byte_pair[1] <= 126
                else f"\\x{byte_pair[1]:02x}"
            )
            print(
                f"  {i+1}. ({byte_pair[0]}, {byte_pair[1]}) -> '{char1}{char2}': {freq}"
            )
        except:
            print(f"  {i+1}. {byte_pair}: {freq}")

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
        pre_frequency_dict: The original pre-tokenization frequency dictionary.
        merge_byte_pair: The byte pair to merge (as a tuple of two bytes).
        merge_index: The index of the merge operation used to decide new token.

    Output:
        new_pre_frequency_dict: The new pre-tokenization frequency dictionary.
    """
    new_pre_frequency_dict = {}
    for token_bytes, frequency in pre_frequency_dict.items():
        token_list = list(token_bytes)
        # Find and merge byte pairs and modify it
        i = 0
        while i < len(token_list) - 1:
            if (token_list[i], token_list[i + 1]) == merge_byte_pair:
                # Replace the pair with new token (256 + merge_index)
                token_list[i] = 256 + merge_index
                # Remove the second byte of the pair
                token_list.pop(i + 1)
            else:
                i += 1
        new_token_key = tuple(token_list)
        new_pre_frequency_dict[new_token_key] = frequency

    return new_pre_frequency_dict


def merge_n_times(initial_pre_frequency_dict: dict, n: int):
    """
    This function merges the most frequent byte pair n times.
    Inputs:
        initial_pre_frequency_dict: The initial pre-tokenization frequency dictionary.
        n: The number of times to merge the most frequent byte pair.

    Output:
        new_pre_frequency_dict: The new pre-tokenization frequency dictionary after merging.
    """
    # @ keep the frequency of current byte pairs by another dictionary
    frequency_tensor = generate_byte_pair_frequency_tensor(initial_pre_frequency_dict)
    # merge_byte_pair is a tuple containing 2 int standing for the byte pair to merge
    merge_byte_pair = (frequency_tensor[0][0], frequency_tensor[0][1])

    # @ first, merge 1 byte pair and generate new byte pair frequency after merge
    # 2 things need to be done if we want to merge the byte pair:
    # 1. Based on the pair to merge, generate new pre-tokenization dict
    # 2. Based on the new pre dict, generate the byte pair frequency dict
    new_pre_frequency_dict = generate_merged_pre_frequency_dict(
        initial_pre_frequency_dict, merge_byte_pair, 0
    )
    # @ rest merge in iteration
    for _ in range(n - 1):
        new_frequency_tensor = generate_byte_pair_frequency_tensor(
            new_pre_frequency_dict
        )
        # merge_byte_pair is a tuple containing 2 int standing for the byte pair to merge
        new_merge_byte_pair = (new_frequency_tensor[0][0], new_frequency_tensor[0][1])

        # We need to use new pre_frequency_dict containing 1st merge as input
        new_new_pre_frequency_dict = generate_merged_pre_frequency_dict(
            new_pre_frequency_dict, new_merge_byte_pair, _ + 1
        )

        new_pre_frequency_dict = new_new_pre_frequency_dict
        # new_new_frequency_tensor = generate_byte_pair_frequency_tensor(
        #     new_new_pre_frequency_dict
        # )
    return new_pre_frequency_dict


f = "../data/valid.txt"
num_processes = 40000

# This is the pattern used to divide words, i.e. pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# boundaries is a list containing the location of boundaries in the txt file
boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
# @ pre-tokenization to the text
# @ pre_frequency_dict with no merge operation done
pre_frequency_dict = generate_pre_frequency_dict(f, PAT, boundaries)
# @ final pre_frequency_dict after merging n times
final_pre_frequency_dict = merge_n_times(pre_frequency_dict, 3)
