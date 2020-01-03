import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm
from glob import glob
import os


class DataGetter(object):
    """
    Helper class to load large .txt files as tensors

    """

    def __init__(self, file_dir: str, tokenizer: str = "bert-base-cased"):

        self.file_dir = file_dir
        print(f"Loading tokenizer: {tokenizer}")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.seqlen = self.tokenizer.max_len

    def load_data(self, filename: str):
        """
        Given a filename, returns that file object
        
        :param filename: the file to be loaded from the self.file_dir directory
        """

        with open(self.file_dir + filename, "r") as file:
            data = file.read()
            file.close()

        data = self.tokenizer.encode(data)
        return data

    def build_dataset(self, files: []):
        """
        Given a list of filenames, returns a TensorDataset of the
        encoded files

        """
        data = []
        for file in tqdm(files):
            data.append(torch.tensor(self.tokenizer.encode(self.load_data(file))))

        return data

    def get_seqdim(self, data: []):
        """
        Given a list of tensors, calculates the longest length amongst them
        and returns the required sequence dimension.

        :param data: list containing tensors
        """

        max_ = 0
        for tensor in data:

            len_ = tensor.shape[0]
            if len_ > max_:
                max_ = len_

        self.seqdim = max_ // self.seqlen + 1

        return max_ // self.seqlen + 1

    def padding_2d(self, data: []):
        """
        Given a list of tensors, and a sequence dimension, pads the tensors to the
        needed dimension and then transforms them to 2D tensors with the same sequence
        dimension and sequence length.
        """
        for row, tensor in enumerate(data):

            to_pad = self.seqdim * self.seqlen - tensor.shape[0]
            tensor = tensor.tolist() + to_pad * [self.tokenizer.pad_token_id]
            tensor = torch.tensor(tensor, dtype=torch.float32)

            tensor = tensor.view(1, self.seqdim, self.seqlen)

            if row == 0:
                tensor_dataset = tensor
            else:
                tensor_dataset = torch.cat((tensor_dataset, tensor), dim=0)

        return tensor_dataset
