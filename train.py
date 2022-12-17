import torch
import torch.nn as nn
from data import CharDataset
from model import CharRNN

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        optimizer.zero_grad()

        # Compute prediction error
        pred = model(X_train)
        loss = loss_fn(pred, y_train.squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_train)
            
            pred = torch.nn.functional.softmax(pred, dim=1)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if batch % 1000 == 0:
            print(sample(model, 300, 'e'))
            model.train()

def sample(model, length, start_char='a', temperature=0.5):
    model.eval()
    with torch.no_grad():
        sample = [start_char]
        sample_ix = [training_set.char_to_ix[start_char]]

        for _ in range(length):
            ix = torch.tensor([sample_ix]).to(device).long()

            pred = model(ix)

            pred = torch.nn.functional.softmax(pred / temperature, dim=1)
            pred_ix = torch.multinomial(pred[0], 1)

            sample.append(training_set.ix_to_char[pred_ix.item()])
            sample_ix.append(pred_ix)
    
    return ''.join(sample)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

params = {
    'batch_size': 1024,
    'shuffle': True,
}

training_set = CharDataset('data.txt', 100)
training_generator = torch.utils.data.DataLoader(training_set, **params)

distinct_characters = len(training_set.chars)
hidden_units = 128
embedding_size = 16

model = CharRNN(distinct_characters, hidden_units, distinct_characters, embedding_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 10
  
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_generator, model, loss_fn, optimizer)

print("Done!")
