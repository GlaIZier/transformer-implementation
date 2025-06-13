from transformer.data import EnFrDataset, Partition, TokEnFrDataset


def test_en_fr_dataset():
    dataset = EnFrDataset("../../data/eng_-french-nano.csv", val_ratio=0.3)
    train_sample = dataset[0]
    dataset.partition = Partition.VAL
    val_sample = dataset[0]
    assert train_sample != val_sample
    print(train_sample)
    assert len(train_sample) == 2
    print(val_sample)
    assert len(val_sample) == 2

    for i, d in enumerate(dataset):
        if i > 2:
            break
        print(d)

def test_tok_en_fr_dataset():
    dataset = TokEnFrDataset("../../data/eng_-french-nano.csv", val_ratio=0.3)
    train_sample = dataset[0]
    dataset.partition = Partition.VAL
    val_sample = dataset[0]
    assert train_sample != val_sample
    print(train_sample)
    assert len(train_sample) == 3
    print(val_sample)
    assert len(val_sample) == 3

    for i, d in enumerate(dataset):
        if i > 2:
            break
        print(d)