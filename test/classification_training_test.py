import torch

from thesis import classification_training as ct

def test_hierarchy_similarity():
    anchors = [
        ['Quick', 'Brown', 'Fox', 'Lazy', 'Dog'],
        ['Quick', 'Brown', 'Fox', 'Lazy', 'Dog'],
        ['Quick', 'Brown', 'Fox'],
        ['Pot', 'Kettle', 'Black'],
        ['Pot', 'Kettle', 'Black'],
        ['Pot', 'Kettle', 'Black'],
    ]
    negatives = [
        ['Quick', 'Brown', 'Fox', 'Lazy', 'Dog'],
        ['Quick', 'Brown', 'Cat', 'Lazy', 'Dog'],
        ['Quick', 'Brown', 'Fox', 'Snoozy', 'Hyena'],
        ['Quick', 'Brown', 'Fox', 'Lazy', 'Dog'],
        ['Pot'],
        ['Hello', 'Darkness', 'My', 'Old', 'Friend']
    ]
    expected = torch.tensor([
        1,
        2/5,
        1, # 1 due to all nodes found in the negative, following Tonioni's Eq 2
        0,
        1/3,
        0
    ], dtype = torch.float)

    actual = ct.hierarchy_similarity(anchors, negatives)
    assert expected.allclose(actual)
