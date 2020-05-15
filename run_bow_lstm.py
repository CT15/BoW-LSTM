import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from BowLSTM import BowLSTM
from WeightedBCELoss import WeightedBCELoss
import utils


DATA_NUM = 5
SEQ_LEN = 20
BATCH_SIZE = 20
INTERVENED_RATIO = 0.25
EPOCHS = 10
CLIP = 5
VAL_EVERY = 200
NAME = 'baseline5_10epochs'

test, train, val = utils.load_test_train_val(1)
vectors, labels = utils.get_vectors_labels(val, utils.get_word2idx())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter(f'runs/{NAME}_val')
writer2 = SummaryWriter(f'runs/{NAME}_train')

word2idx = utils.get_word2idx()
train_loader, test_loader, val_loader = utils.get_train_test_val_loaders(DATA_NUM, SEQ_LEN, BATCH_SIZE, word2idx)

print('DataLoader ... READY!')

criterion = WeightedBCELoss(zero_weight=INTERVENED_RATIO, one_weight=1-INTERVENED_RATIO)
model = BowLSTM(criterion=criterion, input_dim=len(word2idx))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.zero_grad()
model.train()

batch_completed = 0

for epoch in range(1, EPOCHS+1):
    training_loss = 0
    data_completed = 0
    data_multiple = 1

    for inputs, labels, lengths in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        predictions, _ = model(inputs, lengths)
        loss, _, _ = model.loss(predictions, labels, lengths)

        training_loss += loss
        data_completed += len(lengths)
        batch_completed += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        print(f'Batch completed: {batch_completed}')
        
        if data_completed >= data_multiple * VAL_EVERY:
            f1, precision, recall, _, _ = utils.evaluate(model, val_loader, BATCH_SIZE)
            print(f'Evaluating now VALIDATE (f1, precision, recall): {f1}, {precision}, {recall}')
            writer.add_scalar('f1', f1, batch_completed)
            writer.add_scalar('precision', precision, batch_completed)
            writer.add_scalar('recall', recall, batch_completed)
            writer.add_scalar('training_loss', training_loss / data_completed, batch_completed)
            
            f1, precision, recall, _, _ = utils.evaluate(model, train_loader, BATCH_SIZE)
            writer2.add_scalar('f1', f1, batch_completed)
            writer2.add_scalar('precision', precision, batch_completed)
            writer2.add_scalar('recall', recall, batch_completed)
            
            data_multiple += 1

f1, precision, recall, accuracy, conf_matrix = utils.evaluate(model, test_loader, BATCH_SIZE)
print('TEST')
print(f'F1: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'Conf matrix: {conf_matrix}')
