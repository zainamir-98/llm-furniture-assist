from collections import Counter
import torch
from typing import List
from activity_recognition.feature_extractor import feature_extractor, pad_and_stack_vectors
from activity_recognition.non_sequential.dataset_non_seq import HAND_CATEGORY, IRRELAVNT_CLASSES, ActionDataset
from activity_recognition.sequential.dataset_seq import Sequence, SequenceLoader


ACTIVITIES = {
    "take leg": 0,
    "assemble leg": 1,
    "grab drill": 2,
    "use drill": 3,
    "drop drill": 4,
    "take screw driver": 5,
    "use screw driver": 6,
    "drop screw driver": 7
}

INV_ACTIVITIES = {value: key for key, value in ACTIVITIES.items()}


class Ensemble:

    def __init__(self,
                 seq_model,
                 non_seq_model,
                 weight=1):
        self.seq_model = seq_model
        self.non_seq_model = non_seq_model

        # weighting factor for sequential model
        # 1 means the vote distribution remains 3:1
        # if larger than 1 sequential vote is multiplied, i.e.
        # 3 means vote distribution 1:1
        self.weight = weight

        self.seq_output = None # should be int
        self.non_seq_output = None # should be list
        self.sequence_length = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_model.to(self.device)
        self.non_seq_model.to(self.device)


    def vote(self):
        if self.seq_output is not None and self.non_seq_output is not None:
            if self.prob:
                sum_non_seq = torch.tensor(self.non_seq_output).sum(dim=0)
                s = sum_non_seq + torch.tensor(self.seq_output) * self.weight
                s /= self.sequence_length
                majority_vote = INV_ACTIVITIES[torch.argmax(s).item()]
            else:
                occurences = Counter(self.non_seq_output)
                occurences[self.seq_output] = (occurences.get(self.seq_output, 0) + 1)\
                                                * self.weight
                votes = {INV_ACTIVITIES[action_class]: occ for action_class, occ in occurences.items()}
                majority_vote = max(votes, key=votes.get)
            return majority_vote
        else:
            raise Exception("models need to perform prediction first")
        

    def predict(self, root_dir: str, data: List[dict], prob=True):
        self.prob = prob
        if len(data) != self.sequence_length:
            raise Exception(f"Expected {self.sequence_length} frames, got {len(data)}.")
        self.non_seq_output = self._non_seq_prediction(root_dir, data)
        file_names = [d.get("file_name") for d in data]
        self.seq_output = self._seq_prediction(root_dir, file_names)
        return {
            "non-sequential": self.non_seq_output,
            "sequential": self.seq_output
        }


    def _non_seq_prediction(self, root_dir: str, data: List[dict]):        
        dataset = ActionDataset(root_dir,
                                data,
                                actions=None,
                                feature_extractor=feature_extractor,
                                pad=pad_and_stack_vectors,
                                test=True)
        preds = []
        with torch.no_grad():
            self.non_seq_model.eval()
            for i in range(self.sequence_length):
                feature_vector = dataset[i].to(self.device).float()
                out = self.non_seq_model(feature_vector)
                probas = torch.softmax(out, dim=0)
                if self.prob:
                    preds.append(probas.tolist())
                else:
                    idx = torch.argmax(probas).item()
                    preds.append(idx)
        return preds
    

    def _seq_prediction(self, root_dir: str, frames: List[str]):
        seq = Sequence(
            root_dir,
            activity=None,
            frames=frames
        )
        loader = SequenceLoader(
            [seq],
            feature_extractor=feature_extractor,
            test=True
        )
        with torch.no_grad():
            self.seq_model.eval()
            feature_vector = loader[0].to(self.device).float()
            out = self.seq_model(feature_vector.unsqueeze(0))
            probas = torch.softmax(out, dim=1)
            if self.prob:
                return probas.tolist()
            else:
                idx = torch.argmax(probas).item()[0]
                return idx


