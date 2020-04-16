import tsm_mobilenetv2
import torch
import dataset
import time
import copy
import math
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(pretrained_path):
    model = tsm_mobilenetv2.load_model(pretrained_path)

    # freezee all conv-layer and fc-layer
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze fc-layer
    model.classifier.weight.requires_grad = True
    buffer = [torch.zeros([1, 3, 56, 56]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 20, 7, 7]),
              torch.zeros([1, 20, 7, 7])]

    return model, buffer


def weighted_averaging(idx):
    return 1.0/(1+math.exp(-0.2*(idx-8)))


def train(model, buffer, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    softmax = torch.nn.Softmax(1)

    save_freq = 5
    model.eval()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        logging.info(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            dataloader = None
            if phase == 'train':
                dataloader = train_loader
            else:
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            counter = 0
            for inputs, labels in dataloader:
                with torch.set_grad_enabled(phase == 'train'):
                    scores = [0.0 for _ in range(27)]
                    idx = 0
                    video_loss = 0
                    for input in inputs:
                        input = input.squeeze(0).to(device)
                        labels = labels.to(device)

                        output, *buffer = model(input, *buffer)
                        weight = weighted_averaging(idx)
                        video_loss += weight * criterion(output, labels)
                        x = softmax(output).data[0].tolist()
                        for j in range(len(scores)):
                            scores[j] += weight*x[j]
                        idx += 1

                    preds = 0
                    for i in range(len(scores)):
                        if scores[i] > scores[preds]:
                            preds = i

                    avg_video_loss = video_loss / len(inputs)
                    if phase == 'train':
                        optimizer.zero_grad()
                        avg_video_loss.backward()
                        optimizer.step()

                    running_loss += avg_video_loss.item()
                    if preds == labels.data:
                        running_corrects += 1
                # print(f'progress: {counter}/{len(dataloader)}')
                counter += 1

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)

            loss_acc_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)
            print(loss_acc_info)
            logging.info(loss_acc_info)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        if epoch % save_freq == 0:
            torch.save(best_model_wts, f'{time.time()}.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    best_val_acc_info = 'Best val Acc: {:4f}'.format(best_acc)
    print(best_val_acc_info)
    logging.info(best_val_acc_info)

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def create_optimizer(model):
    params_to_update = []
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    return torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9)


if __name__ == "__main__":

    # setup dataset
    from preprocess import Preprocess
    directory = '/home/ds/Data/academic/dataset_v2'
    transform = Preprocess.get_transform()
    loader = dataset.VideoLoader(directory, transform)
    train_loader = loader.get_train_loader(batch_size=1)
    val_loader = loader.get_val_loader(batch_size=1)

    # setup model
    model, buffer = initialize_model("pretrained.pth.tar")

    # setup trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    # setup logger
    logging.basicConfig(level=logging.INFO, filename='training_log.txt')

    model = model.to(device)
    buffer = [b.to(device) for b in buffer]
    model, hist = train(model, buffer, train_loader,
                        val_loader, criterion, optimizer, num_epochs=5)

    torch.save(model.state_dict(), 'result.pth')
    with open('val_history.txt', 'w') as f:
        for item in hist:
            f.write(str(item))
