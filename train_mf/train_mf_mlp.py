import os
import torch
import argparse
import torch.nn as nn
from torch.optim import AdamW
from mf_mlp import TradModel
from utils_mf_mlp import DataLoader, Batchify, now_time, root_mean_square_error, mean_absolute_error

parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('-data_path', '--data_path', type=str, default="./data/TripAdvisor/reviews.pickle",
                    help='path for loading the pickle data')
parser.add_argument('-index_dir', '--index_dir', type=str, default="./data/TripAdvisor/1/",
                    help='load indexes')
parser.add_argument('-model_name', '--model_name', type=str, default="mf",
                    help='model name')
parser.add_argument('-emsize', '--emsize', type=int, default=64,
                    help='dim of embedding')
parser.add_argument('-lr', '--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('-epochs', '--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('-batch_size', '--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('-seed', '--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('-cuda', '--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('-log_interval', '--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('-checkpoint', '--checkpoint', type=str, default='./mf/',
                    help='directory to save the final model')
parser.add_argument('-endure_times', '--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
unk = '<unk>'
corpus = DataLoader(args.data_path, args.index_dir)
train_data = Batchify(corpus.train, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, args.batch_size)
test_data = Batchify(corpus.test, args.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
model = TradModel(nuser, nitem, args.emsize, args.model_name)
model.to(device)
rating_criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=args.lr)


###############################################################################
# Training code
###############################################################################


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating = data.next_batch()  # data.step += 1
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        rating_p = model(user, item)
        r_loss = rating_criterion(rating_p, rating)
        loss = r_loss
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                cur_r_loss, data.step, data.total_step))
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            rating_p = model(user, item)
            r_loss = rating_criterion(rating_p, rating)

            batch_size = user.size(0)
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return rating_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating_p = model(user, item)
            rating_predict.extend(rating_p.tolist())

            if data.step == data.total_step:
                break
    return rating_predict


print(now_time() + 'Tuning Prompt with LoRA')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_r_loss = evaluate(val_data)
    val_loss = val_r_loss
    print(now_time() + 'rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        val_r_loss, val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# # Finally, freeze all parameters except for LoRA parameters.
# for name, param in model.named_parameters():
#     param.requires_grad = True

# Run on test data.
test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'rating loss {:4.4f} on test | End of training'.format(
    test_r_loss))
print(now_time() + 'Generating text')
rating_predicted = generate(test_data)
# rating
predicted_rating = [(r, p) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))
