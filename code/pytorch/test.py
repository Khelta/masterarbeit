from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.data import evaluate_print

from torch import flatten

from datasets import prepare_data

train_loader, test_loader = prepare_data("mnist", 0, 0.25)
data_labels = []
images = []
for batch, labels in train_loader:
    data_labels += labels.tolist()
    for img in batch:
        flat = flatten(img)
        images.append(flat.tolist())

clf = DeepSVDD(batch_size=256)
clf.fit(images)

decision_scores = clf.decision_scores_

for i in range(0, len(decision_scores)):
    print(data_labels[i], decision_scores[i])
