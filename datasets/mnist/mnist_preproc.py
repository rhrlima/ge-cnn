import numpy as np
import os
import pickle
import zipfile

root_folder = ''

zip_file = 'mnist-csv.zip'
train_file = 'mnist_train'
test_file = 'mnist_test'
pickle_file = 'mnist.pickle'


def maybe_extract(zip_file):
    file_path = os.path.join(root_folder, zip_file)
    with zipfile.ZipFile(file_path) as zip_ref:
        zip_ref.extractall(root_folder)
    print(f'extracted \"{zip_file}\" file to \"{root_folder}\" folder.')


def load_dataset_from_csv(file, min_images=None, force=False):

    pixel_depth = 255

    file += '.csv'
    file_path = os.path.join(root_folder, file)
    with open(file_path) as f:
        f.readline()  # skip headers

        dataset = []
        labels = []
        for i, line in enumerate(f):
            line = line.replace('\n', '').split(',')
            label = line[0]
            img_data = np.array(line[1:], dtype=float) / pixel_depth
            dataset.append(img_data)
            labels.append(label)

        dataset = np.asarray(dataset, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        dataset = dataset.reshape(-1, 28, 28)
        print('dataset ', dataset.shape)
        print('labels ', labels.shape)
        print('mean ', np.mean(dataset))
        print('std ', np.std(dataset))

    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def split_dataset(dataset, labels, split_size):
    valid_dataset = dataset[:split_size, :, :]
    train_dataset = dataset[split_size:, :, :]
    valid_labels = labels[:split_size]
    train_labels = labels[split_size:]
    return train_dataset, train_labels, valid_dataset, valid_labels


if __name__ == '__main__':

    maybe_extract(zip_file)

    train_dataset, train_labels = load_dataset_from_csv(train_file)
    test_dataset, test_labels = load_dataset_from_csv(test_file)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(
        train_dataset, train_labels, 10000)

    print(train_dataset.shape, train_labels.shape)
    print(valid_dataset.shape, valid_labels.shape)
    print(test_dataset.shape, test_labels.shape)

    pickle_file = os.path.join(root_folder, pickle_file)
    with open(pickle_file, 'wb') as f:
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            'input_shape': (28, 28, 1),
            'num_classes': 10
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

    stat_info = os.stat(pickle_file)
    print('Compressed pickle size: ', stat_info.st_size)
