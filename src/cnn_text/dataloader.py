import numpy as np
from string import punctuation
import torch
from torch.utils.data import TensorDataset, DataLoader


def read_data(samples_num=100):
    # read data from text files
    with open('data/reviews/reviews.txt', 'r') as f:
        reviews = f.read()
    with open('data/reviews/labels.txt', 'r') as f:
        labels = f.read()

    # print some example review/sentiment text
    print(reviews[:1000])
    print()
    print(labels[:20])

    # get rid of punctuation
    reviews = reviews.lower()  # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')

    # all_text = ' '.join(reviews_split)

    # create a list of all words
    # all_words = all_text.split()

    # 1=positive, 0=negative label conversion
    labels_split = labels.split('\n')
    # debug
    reviews_split = reviews_split[:samples_num]
    labels_split = labels_split[:samples_num]

    return reviews_split, labels_split


class Loader():
    def __init__(self, embed_model, seq_length=200, batch_size=50, split_frac=0.8):
        self.embed_model = embed_model

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.split_frac = split_frac

    def process_data(self, reviews_split, labels_split):
        encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

        print('Number of reviews before removing outliers: ', len(reviews_split))

        ## remove any reviews/labels with zero length from the reviews_ints list.

        # get indices of any reviews with length 0
        non_zero_idx = [ii for ii, review in enumerate(reviews_split) if len(review.split()) != 0]

        # remove 0-length reviews and their labels
        reviews_split = [reviews_split[ii] for ii in non_zero_idx]
        encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

        print('Number of reviews after removing outliers: ', len(reviews_split))

        return reviews_split, encoded_labels

    # convert reviews to tokens
    def tokenize_all_reviews(self, embed_model, reviews_split):
        # split each review into a list of words
        reviews_words = [review.split() for review in reviews_split]

        tokenized_reviews = []
        for review in reviews_words:
            ints = []
            for word in review:
                try:
                    idx = embed_model.vocab[word].index
                except:
                    idx = 0
                ints.append(idx)
            tokenized_reviews.append(ints)

        return tokenized_reviews

    def pad_features(self, reviews_split):
        # tokenize
        tokenized_reviews = self.tokenize_all_reviews(self.embed_model, reviews_split)

        # get features
        features = np.zeros((len(tokenized_reviews), self.seq_length), dtype=int)

        ## test statements - do not change - ##
        assert len(features) == len(tokenized_reviews), "Features should have as many rows as reviews."
        assert len(features[0]) == self.seq_length, "Each feature row should contain seq_length values."

        # print first 8 values of the first 20 batches
        print(features[:20, :8])

        return features

    def split(self, features, encoded_labels):
        ## split data into training, validation, and test data (features and labels, x and y)

        split_idx = int(len(features) * self.split_frac)
        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

        test_idx = int(len(remaining_x) * 0.5)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

        ## print out the shapes of your resultant feature data
        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              "\nValidation set: \t{}".format(val_x.shape),
              "\nTest set: \t\t{}".format(test_x.shape))

        return train_x, train_y, val_x, val_y, test_x, test_y

    def create_tensor(self, train_x, train_y, val_x, val_y, test_x, test_y):
        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # dataloaders
        # shuffling and batching data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)
        return train_loader, valid_loader, test_loader

    def load(self, reviews, labels):
        reviews_split, encoded_labels = self.process_data(reviews, labels)
        features = self.pad_features(reviews_split)
        train_x, train_y, val_x, val_y, test_x, test_y = self.split(features, encoded_labels)
        train_loader, valid_loader, test_loader = self.create_tensor(train_x, train_y, val_x, val_y, test_x, test_y)

        return train_loader, valid_loader, test_loader

# def main():
#     loader = Loader()
#     loader.load(reviews, labels)
#
#
# if __name__ == '__main__':
#     main()
