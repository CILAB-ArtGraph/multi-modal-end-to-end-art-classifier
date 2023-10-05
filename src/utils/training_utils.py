import copy
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


def fine_tune(model, graph, train_loader, validation_loader,
              criterions, optimizer, early_stop, scheduler,
              num_epochs=100, device=torch.device('cuda:0')):

    best_epoch = 0
    best_loss = None
    style_criterion, genre_criterion, emotion_criterion = criterions
    for epoch in range(1, num_epochs + 1):
        if early_stop.stop:
            break
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 120)

        data_loader = None
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader
            running_loss = 0.0
            running_corrects_style = running_corrects_genre = running_corrects_emotion = 0

            for _, image_features, (style_labels, genre_labels, emotion_labels) in tqdm(data_loader):
                image_features = image_features.to(device, non_blocking=True)
                style_labels = style_labels.to(device, non_blocking=True)
                genre_labels = genre_labels.to(device, non_blocking=True)
                emotion_labels = emotion_labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                out_style, out_genre, out_emotion = model(image_features, graph.x_dict, graph.edge_index_dict)

                _, preds_style = torch.max(nn.Softmax(dim=1)(out_style), dim=1)
                _, preds_genre = torch.max(nn.Softmax(dim=1)(out_genre), dim=1)
                _, preds_emotion = torch.max(nn.Softmax(dim=1)(out_emotion), dim=1)

                loss_style = style_criterion(out_style, style_labels)
                loss_genre = genre_criterion(out_genre, genre_labels)
                loss_emotion = emotion_criterion(out_emotion, emotion_labels)

                loss = (loss_style + loss_genre + loss_emotion)/3

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * data_loader.batch_size
                running_corrects_style += torch.sum(preds_style == style_labels)
                running_corrects_genre += torch.sum(preds_genre == genre_labels)
                running_corrects_emotion += torch.sum(preds_emotion == emotion_labels)

            epoch_loss = running_loss / (len(data_loader) * data_loader.batch_size)
            epoch_acc_style = running_corrects_style.double() / (len(data_loader) * data_loader.batch_size)
            epoch_acc_genre = running_corrects_genre.double() / (len(data_loader) * data_loader.batch_size)
            epoch_acc_emotion = running_corrects_emotion.double() / (len(data_loader) * data_loader.batch_size)

            print(f'''{phase} Loss: {epoch_loss:.4f} Style Acc: {epoch_acc_style:.4f}
            Genre Acc: {epoch_acc_genre:.4f}
            Emotion Acc: {epoch_acc_emotion:.4f}''')

            if phase == 'val':
                scheduler.step(epoch_loss)
                early_stop(epoch_loss, model)
                if epoch == 1 or best_loss > epoch_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch

    print(f'Best epoch: {best_epoch:04d}')
    print(f'Best loss: {best_loss:.4f}')


def fine_tune_single_task(model, graph, train_loader, validation_loader,
              criterion, optimizer, early_stop, scheduler,
              num_epochs=100, device=torch.device('cuda:0')):

    best_epoch = 0
    best_loss = None
    for epoch in range(1, num_epochs + 1):
        if early_stop.stop:
            break
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 120)

        data_loader = None
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader
            running_loss = 0.0
            running_corrects = 0

            for _, image_features, labels in tqdm(data_loader):
                image_features = image_features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                out = model(graph.x_dict, graph.edge_index_dict, image_features)

                _, preds = torch.max(nn.Softmax(dim=1)(out), dim=1)

                loss = criterion(out, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * data_loader.batch_size
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / (len(data_loader) * data_loader.batch_size)
            epoch_acc = running_corrects.double() / (len(data_loader) * data_loader.batch_size)

            print(f'''{phase} Loss: {epoch_loss:.4f} Style Acc: {epoch_acc:.4f}''')

            if phase == 'val':
                scheduler.step(epoch_loss)
                early_stop(epoch_loss, model)
                if epoch == 1 or best_loss > epoch_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch

    print(f'Best epoch: {best_epoch:04d}')
    print(f'Best loss: {best_loss:.4f}')


def add_predictions(graph, idx, preds):
    preds = {k: torch.argmax(v, dim=1) for k, v in preds.items()}
    styles = torch.vstack([idx.cpu(), preds['style'].cpu()+1]) if 'style' in preds else None
    genres = torch.vstack([idx.cpu(), preds['genre'].cpu()]) if 'genre' in preds else None
    emotions = torch.vstack([idx.cpu(), preds['emotion'].cpu()]) if 'emotion' in preds else None
    if styles is not None:
        graph['artwork', 'hasstyle','style'].edge_index = torch.hstack([graph['artwork', 'hasstyle', 'style'].edge_index.cpu(), styles]).type(
        torch.LongTensor)
    if genres is not None:
        graph['artwork', 'hasgenre','genre'].edge_index = torch.hstack([graph['artwork', 'hasgenre','genre'].edge_index.cpu(), genres]).type(
        torch.LongTensor)
    if emotions is not None:
        graph['artwork', 'elicit','emotion'].edge_index = torch.hstack([graph['artwork', 'elicit','emotion'].edge_index.cpu(), emotions]).type(
        torch.LongTensor)
    return graph


def test_single_task(model, data_loader, graph, update_graph = False, device=torch.device('cuda:0')):
    tot_pred = None
    tot_lab = []
    test_graph = copy.deepcopy(graph).to(device)

    for idx, images, labels in tqdm(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        pred = model(test_graph.x_dict, test_graph.edge_index_dict, images)

        if update_graph:
            if pred.shape[1] == 30:  # style
                preds = {'style': pred}
            elif pred.shape[1] == 18:  # genre
                preds = {'genre': pred}
            else:  # emotion
                preds = {'emotion': pred}
            test_graph = add_predictions(graph, idx, preds).to(device)

        if tot_pred is None:
            tot_pred = pred.cpu()
        else:
            tot_pred = torch.vstack([tot_pred, pred.cpu()])
        tot_lab += labels.cpu().tolist()

    return tot_pred, tot_lab



def test(model, data_loader, graph, update_graph=False, device=torch.device('cuda:0')):
    tot_pred = {t: None for t in ['style', 'genre', 'emotion']}
    tot_lab = {t: [] for t in ['style', 'genre', 'emotion']}
    test_graph = copy.deepcopy(graph).to(device)
    for idx, images, (style_labels, genre_labels, emotion_labels) in tqdm(data_loader):
        images = images.to(device, non_blocking=True)
        style_labels = style_labels.to(device, non_blocking=True)
        genre_labels = genre_labels.to(device, non_blocking=True)
        emotion_labels = emotion_labels.to(device, non_blocking=True)

        style_pred, genre_pred, emotion_pred = model(images, test_graph.x_dict, test_graph.edge_index_dict)

        if update_graph:
            preds = {'style': style_pred,
                     'genre': genre_pred,
                     'emotion': emotion_pred}
            test_graph = add_predictions(graph, idx, preds).to(device)

        if tot_pred['style'] is None:
            tot_pred['style'] = style_pred.cpu()
            tot_pred['genre'] = genre_pred.cpu()
            tot_pred['emotion'] = emotion_pred.cpu()
        else:
            tot_pred['style'] = torch.vstack([tot_pred['style'], style_pred.cpu()])
            tot_pred['genre'] = torch.vstack([tot_pred['genre'], genre_pred.cpu()])
            tot_pred['emotion'] = torch.vstack([tot_pred['emotion'], emotion_pred.cpu()])

        tot_lab['style'] += style_labels.cpu().tolist()
        tot_lab['genre'] += genre_labels.cpu().tolist()
        tot_lab['emotion'] += emotion_labels.cpu().tolist()

    return tot_pred, tot_lab


def compute_topk(true, pred, k):
    # target data frame to compute topk
    df = pd.DataFrame(true, columns=['true'])
    # useful dataframe to compute top k target for each artwork
    temp = pd.DataFrame(pd.DataFrame(pred).apply(lambda x: x.tolist(), axis=1), columns=['pred'])
    temp['temp'] = temp.pred.map(lambda x: list(range(len(x))))  # for each artwork list of classes
    temp['temp'] = temp.apply(lambda x: list(zip(x['pred'], x['temp'])), axis=1)  # zip probabilities with classes
    temp['topk'] = temp['temp'].map(lambda x: list(sorted(x, reverse=True))[:k])  # sorting
    df['topk'] = temp.topk.map(lambda x: list(map(lambda y: y[1], x)))  # taking only classes

    df['cond'] = df.apply(lambda x: x['true'] in x['topk'], axis=1)
    return df[df.cond].index.shape[0] / df.index.shape[0]


# plot confusion matrix
def plot_confusion_matrix(true, pred, task=None, strategy=None, labels=None, hop=1):
    conf_mat = confusion_matrix(true, pred, normalize='true')
    sns.set(font_scale=1.5)  # for label size
    fig, ax = plt.subplots(figsize=(12, 12))

    ax = sns.heatmap(pd.DataFrame(conf_mat), annot=False, cmap=plt.cm.Blues,
                     xticklabels=labels,
                     yticklabels=labels,
                     square=True,
                     linewidths=.50)

    plt.show()
    if task:
        fig.savefig(f'{task}_{strategy}.svg')