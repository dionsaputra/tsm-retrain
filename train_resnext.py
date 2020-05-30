import tsm_mobilenetv2
import torch
import dataset
import time
import copy
import math
import logging
import resnext

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(pretrained_path):
    torch_module = resnext.resnet101(sample_size=112, sample_duration=16, num_classes=27)
    state_dict = torch.load(pretrained_path)["state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    torch_module.load_state_dict(new_state_dict)

    # freeze feature-extraction
    for param in torch_module.parameters():
        param.requires_grad = False

    # unfreeze fc-layer
    torch_module.fc.weight.requires_grad = True

    return torch_module


def weighted_averaging(idx):
    return 1.0/(1+math.exp(-0.2*(idx-8)))


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    softmax = torch.nn.Softmax(1)

    save_freq = 1
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
                    inputs_3d = torch.Tensor(16, 1, 3, 112, 112)
                    torch.cat(inputs, out=inputs_3d)
                    inputs_3d = inputs_3d.permute(1, 2, 0, 3, 4)
                    inputs_3d = inputs_3d.to(device)
                    labels = labels.to(device)

                    output = model(inputs_3d)
                    loss = criterion(output, labels)

                    output = output.data[0].tolist()
                    preds = output.index(max(output))

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
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


def test(model, buffer, dataloader):
    softmax = torch.nn.Softmax(1)
    model.eval()
    running_corrects = 0
    counter = 0
    for inputs, labels in dataloader:
        scores = [0.0 for _ in range(27)]
        idx = 0
        for input in inputs:
            input = input.squeeze(0).to(device)
            labels = labels.to(device)

            output, *buffer = model(input, *buffer)
            weight = weighted_averaging(idx)
            x = softmax(output).data[0].tolist()
            for j in range(len(scores)):
                scores[j] += weight*x[j]
            idx += 1

        preds = 0
        for i in range(len(scores)):
            if scores[i] > scores[preds]:
                preds = i

        if preds == labels.data:
            running_corrects += 1
        counter += 1
        print('{:4f}, {}'.format(running_corrects/counter,
                                 counter*100/len(dataloader.dataset)), end="\r")

    epoch_acc = running_corrects / len(dataloader.dataset)

    loss_acc_info = 'Acc: {:.4f}'.format(epoch_acc)
    print(loss_acc_info)
    # logging.info(loss_acc_info)


def create_optimizer(model):
    params_to_update = []
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    return torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9, weight_decay=0.001)

def test(model, dataloader):
    softmax = torch.nn.Softmax(1)
    model = model.eval()
    running_corrects = 0
    counter = 0

    for inputs, label in dataloader:    
        inputs_3d = torch.Tensor(16, 1, 3, 112, 112)
        torch.cat(inputs, out=inputs_3d)
        inputs_3d = inputs_3d.permute(1, 2, 0, 3, 4)
        inputs_3d = inputs_3d.to(device)

        output = model(inputs_3d)
        output = output.data[0].tolist()
        print(output)
        # print(output.index(max(output)))

        # x = softmax(output).data[0].tolist()
        # print(x)
        # pred = x.index(max(x))
        # print(pred, label[0])

    # for inputs, labels in dataloader:
    #     scores = [0.0 for _ in range(27)]
    #     idx = 0
    #     for input in inputs:
    #         input = input.squeeze(0).to(device)
    #         labels = labels.to(device)

    #         output, *buffer = model(input, *buffer)
    #         weight = weighted_averaging(idx)
    #         x = softmax(output).data[0].tolist()
    #         for j in range(len(scores)):
    #             scores[j] += weight*x[j]
    #         idx += 1

    #     preds = 0
    #     for i in range(len(scores)):
    #         if scores[i] > scores[preds]:
    #             preds = i

    #     if preds == labels.data:
    #         running_corrects += 1
    #     counter += 1
    #     print('{:4f}, {}'.format(running_corrects/counter,
    #                              counter*100/len(dataloader.dataset)), end="\r")

    # epoch_acc = running_corrects / len(dataloader.dataset)

    # loss_acc_info = 'Acc: {:.4f}'.format(epoch_acc)
    # print(loss_acc_info)
    # logging.info(loss_acc_info)


if __name__ == "__main__":

    train_mode = True

    # setup dataset
    from preprocess import Preprocess
    directory = '/home/ds/Data/academic/dataset_v2'
    transform = Preprocess.get_transform(112)
    loader = dataset.VideoLoader(directory, transform)
    train_loader = loader.get_train_loader(batch_size=1)
    val_loader = loader.get_val_loader(batch_size=1)
    test_loader = loader.get_test_loader(batch_size=1)

    # setup model
    model = initialize_model("/home/ds/Data/academic/shared_models_v1/models/jester_resnext_101_RGB_32.pth")
    model = model.to(device)

    # setup trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    # setup logger
    logging.basicConfig(level=logging.INFO, filename='training_log_kopuklu.txt')

    if train_mode:
        model, hist = train(model, train_loader,
                            val_loader, criterion, optimizer, num_epochs=100)

        torch.save(model.state_dict(), 'result_kopuklu.pth')
        with open('val_history1.txt', 'w') as f:
            for item in hist:
                f.write(str(item))
    else:
        #test(model, buffer, test_loader)
        pass
