import torch

class LAWSUMMDataset:
    def __init__(self, data, tokenizer, label = None, max_len):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())

        ids = self.tokenizer.sentence_to_features(
            data[item]
        )

        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(self.label[item], dtype=torch.float),
        }

    def _getTransformerItem_(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())

        inputs = self.tokenizer.encode_plus(
            data,
            None,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.label[item], dtype=torch.float),
        }