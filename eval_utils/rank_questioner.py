import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import options
import visdial.metrics as metrics
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range


def rankQBot(qBot, dataset, split, exampleLimit=None, verbose=0):
    '''
        Evaluates Q-Bot performance on image retrieval when it is shown
        ground truth captions, questions and answers. Q-Bot does not
        generate dialog in this setting - it only encodes ground truth
        captions and dialog in order to perform image retrieval by
        predicting FC-7 image features after each round of dialog.

        Arguments:
            qBot    : Q-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
    '''
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]//4
    else:
        numExamples = exampleLimit//4
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    # enumerate all gt features and all predicted features
    gtImgFeatures = []
    # caption + dialog rounds
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
    logProbsAll = [[] for _ in range(numRounds)]
    featLossAll = [[] for _ in range(numRounds + 1)]
    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        else:
            batch = {
                key: v.contiguous()
                for key, v in batch.items() if hasattr(v, 'cuda')
            }
        with torch.no_grad():  # Equivalent to `volatile=True`
            caption = Variable(batch['cap'])
            captionLens = Variable(batch['cap_len'])
            gtQuestions = Variable(batch['ques'])
            gtQuesLens = Variable(batch['ques_len'])
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])
            gtFeatures = Variable(batch['img_feat'])
        qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)
        predFeatures = qBot.predictImage()
        # Evaluating round 0 feature regression network
        featLoss = F.mse_loss(predFeatures, gtFeatures)
        featLossAll[0].append(torch.mean(featLoss))
        # Keeping round 0 predictions
        roundwiseFeaturePreds[0].append(predFeatures)
        for round in range(numRounds):
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            qBot.observe(
                round, ans=answers[:, round], ansLens=ansLens[:, round])
            logProbsCurrent = qBot.forward()
            # Evaluating logProbs for cross entropy
            logProbsAll[round].append(
                utils.maskedNll(logProbsCurrent,
                                gtQuestions[:, round].contiguous()))
            predFeatures = qBot.predictImage()
            # Evaluating feature regression network
            featLoss = F.mse_loss(predFeatures, gtFeatures)
            featLossAll[round + 1].append(torch.mean(featLoss))
            # Keeping predictions
            roundwiseFeaturePreds[round + 1].append(predFeatures)
            # In ra dự đoán của Q-BOT tại vòng hiện tại (sau khi quan sát Q&A của vòng đó)
            # print(f"Round {round + 1} - Predicted Features (Q-BOT): {predFeatures[0][:5]}")

        gtImgFeatures.append(gtFeatures)

        end_t = timer()
        delta_t = " Time: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []
    poolSize = len(dataset)//4

    # Keeping tracking of feature regression loss and CE logprobs
    # logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll]
    logProbsAll_filtered = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll if all(tensor.dim() > 0 for tensor in lprobs)]
    # featLossAll = [torch.cat(floss, 0).mean() for floss in featLossAll]
    featLossAll_filtered = [torch.cat(floss, 0).mean() for floss in featLossAll if all(tensor.dim() > 0 for tensor in floss)]

    # roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy()
    # roundwiseFeatLoss = torch.cat(featLossAll, 0).data.cpu().numpy()
    # logProbsMean = roundwiseLogProbs.mean()
    # featLossMean = roundwiseFeatLoss.mean()

    # Check if the filtered lists are empty
    if logProbsAll_filtered:
        # Concatenate tensors and convert to numpy
        roundwiseLogProbs = torch.cat(logProbsAll_filtered, 0).data.cpu().numpy()
        logProbsMean = roundwiseLogProbs.mean()
    else:
        # Handle case where logProbsAll_filtered is empty
        logProbsMean = 0.0
        roundwiseLogProbs = None

    if featLossAll_filtered:
        # Concatenate tensors and convert to numpy
        roundwiseFeatLoss = torch.cat(featLossAll_filtered, 0).data.cpu().numpy()
        featLossMean = roundwiseFeatLoss.mean()
    else:
        # Handle case where featLossAll_filtered is empty
        featLossMean = 0.0
        roundwiseFeatLoss = None

    if verbose:
        print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        # num_examples x num_examples
        dists = pairwise_distances(predFeatures, gtFeatures)
        ranks = []
        for i in range(dists.shape[0]):
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        if verbose:
            print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank

        # rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        # Check if roundwiseFeatLoss has been assigned a value
        if roundwiseFeatLoss is not None:
            rankMetrics['featLoss'] = roundwiseFeatLoss[round]
        else:
            # Handle case where roundwiseFeatLoss is not assigned a value
            rankMetrics['featLoss'] = 0.0

        # if round < len(roundwiseLogProbs):
        #     rankMetrics['logProbs'] = roundwiseLogProbs[round]
        # Check if roundwiseLogProbs has been assigned a value
        if roundwiseLogProbs is not None:
            if round < len(roundwiseLogProbs):
                rankMetrics['featLoss'] = roundwiseLogProbs[round]
            else:
                # Handle case where round is out of bounds
                rankMetrics['featLoss'] = 0.0
        else:
            # Handle case where roundwiseLogProbs is not assigned a value
            rankMetrics['featLoss'] = 0.0
            
        rankMetricsRounds.append(rankMetrics)

    rankMetricsRounds[-1]['logProbsMean'] = logProbsMean
    rankMetricsRounds[-1]['featLossMean'] = featLossMean

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds


def rankQABots(qBot, aBot, dataset, split, exampleLimit=None, beamSize=1):
    '''
        Evaluates Q-Bot and A-Bot performance on image retrieval where
        both agents must converse with each other without any ground truth
        dialog. The common caption shown to both agents is not the ground
        truth caption, but is instead a caption generated (pre-computed)
        by a pre-trained captioning model (neuraltalk2).

        Arguments:
            qBot    : Q-Bot
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            exampleLimit : Maximum number of data points to use from
                           the dataset split. If None, all data points.
            beamSize     : Beam search width for generating utterrances
    '''

    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]//4
    else:
        numExamples = exampleLimit//4
    numBatches = (numExamples - 1) // batchSize + 1
    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    gtImgFeatures = []
    roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]

    start_t = timer()
    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {key: v.cuda() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}
        else:
            batch = {key: v.contiguous() for key, v in batch.items() \
                                            if hasattr(v, 'cuda')}
        
        with torch.no_grad():
            caption = Variable(batch['cap'])
            captionLens = Variable(batch['cap_len'])
            gtQuestions = Variable(batch['ques'])
            gtQuesLens = Variable(batch['ques_len'])
            answers = Variable(batch['ans'])
            ansLens = Variable(batch['ans_len'])
            gtFeatures = Variable(batch['img_feat'])
            image = Variable(batch['img_feat'])

        aBot.eval(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        qBot.eval(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

        predFeatures = qBot.predictImage()
        roundwiseFeaturePreds[0].append(predFeatures)

        for round in range(numRounds):
            questions, quesLens = qBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(
                inference='greedy', beamSize=beamSize)
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)
            predFeatures = qBot.predictImage()
            roundwiseFeaturePreds[round + 1].append(predFeatures)
        gtImgFeatures.append(gtFeatures)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Qbot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()
    sys.stdout.write("\n")

    gtFeatures = torch.cat(gtImgFeatures, 0).data.cpu().numpy()
    rankMetricsRounds = []

    print("Percentile mean rank (round, mean, low, high)")
    for round in range(numRounds + 1):
        predFeatures = torch.cat(roundwiseFeaturePreds[round],
                                 0).data.cpu().numpy()
        dists = pairwise_distances(predFeatures, gtFeatures)
        # num_examples x num_examples
        ranks = []
        for i in range(dists.shape[0]):
            # Computing rank of i-th prediction vs all images in split
            rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
        # assert len(ranks) == len(numExamples)
        poolSize = len(ranks)
        meanRank = ranks.mean()
        se = ranks.std() / np.sqrt(poolSize)
        meanPercRank = 100 * (1 - (meanRank / poolSize))
        percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
        percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
        print((round, meanPercRank, percRankLow, percRankHigh))
        rankMetrics['percentile'] = meanPercRank
        rankMetricsRounds.append(rankMetrics)

    dataset.split = original_split
    return rankMetricsRounds[-1], rankMetricsRounds
