import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken


block_size = 8
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 100000
context_size = 2
embedding_dim = 5

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def make_ngrams(train_data):
    ngrams = [
        (
            train_data[i - context_size:i],
            train_data[i]
        ) for i in range(context_size, len(train_data)//100)
    ]
    return ngrams

def train(train_data, vocab_size):
    ngrams = make_ngrams(train_data)
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(vocab_size, embedding_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(50):
        total_loss = 0
        for i, (context, target) in enumerate(ngrams):

            if i % eval_interval == 0:
                print(f"percent done w/ epoch: {i/len(ngrams)}")
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = context

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, target.view(1))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)

        print(f"epoch {epoch} loss: {total_loss}")
    print(losses)  # The loss decreased every iteration over the training data!

    return model, optimizer, losses

class MyEncoder:
    def __init__(self, corpus):
        self.enc = tiktoken.get_encoding("cl100k_base")

        old_tokens = self.enc.encode(corpus)

        self.old2new = {x:i for i, x in enumerate(set(old_tokens))}
        self.new2old = {self.old2new[i]:i for i in self.old2new.keys()}
        self.tokens = [self.old2new[x] for x in old_tokens]
        self.vocab_size = len(set(self.tokens))

    def encode(self, s):
        return [self.old2new[x] for x in self.enc.encode(s)]

    def decode(self, s):
        return self.enc.decode([self.new2old[x] for x in s])
