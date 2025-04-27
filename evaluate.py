import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.dialog_generate import dialogDump
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot, rankQABots
from utils import utilities as utils
from utils.visualize import VisdomVisualize

if __name__ == '__main__':
    # Read command line options
    params = options.readCommandLine()

    # Seed RNG for reproducibility
    manualSeed = 1234
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if params['useGPU']:
        torch.cuda.manual_seed_all(manualSeed)

    # Setup dataloader with reduced batch size
    dlparams = params.copy()
    dlparams['useIm'] = True
    dlparams['useHistory'] = True
    dlparams['numRounds'] = 10
    dlparams['batchSize'] = 8  # Adjust batch size as needed
    splits = ['val', 'test']

    dataset = VisDialDataset(dlparams, splits)
    
    # Transfer dataset parameters
    transfer = ['vocabSize', 'numOptions', 'numRounds']
    for key in transfer:
        if hasattr(dataset, key):
            params[key] = getattr(dataset, key)

    if 'numRounds' not in params:
        params['numRounds'] = 10

    # Always load checkpoint parameters with continue flag
    params['continue'] = True

    excludeParams = ['batchSize', 'visdomEnv', 'startFrom', 'qstartFrom', 'trainMode', \
        'evalModeList', 'inputImg', 'inputQues', 'inputJson', 'evalTitle', 'beamSize', \
        'enableVisdom', 'visdomServer', 'visdomServerPort']

    aBot = None
    qBot = None

    # Load A-Bot
    if params['startFrom']:
        aBot, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=True)
        assert aBot.encoder.vocabSize == dataset.vocabSize, "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aBot.eval()

    # Retain certain dataloader parameters
    for key in excludeParams:
        params[key] = dlparams[key]

    # Load Q-Bot
    if params['qstartFrom']:
        qBot, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
        assert qBot.encoder.vocabSize == params['vocabSize'], "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        qBot.eval()

    # Retain certain dataloader parameters
    for key in excludeParams:
        params[key] = dlparams[key]

    # Plotting on Visdom
    viz = VisdomVisualize(
        enable=bool(params['enableVisdom']),
        env_name=params['visdomEnv'],
        server=params['visdomServer'],
        port=params['visdomServerPort'])
    pprint.pprint(params)
    viz.addText(pprint.pformat(params, indent=4))
    print("Running evaluation!")

    numRounds = params['numRounds']
    if 'ckpt_iterid' in params:
        iterId = params['ckpt_iterid'] + 1
    else:
        iterId = -1

    if 'test' in splits:
        split = 'test'
        splitName = 'test - {}'.format(params['evalTitle'])
    else:
        split = 'val'
        splitName = 'full Val - {}'.format(params['evalTitle'])

    print("Using split %s" % split)
    dataset.split = split

    # Evaluate A-Bot ranking
    if 'ABotRank' in params['evalModeList']:
        print("Performing ABotRank evaluation")
        rankMetrics = rankABot(
            aBot, dataset, split, scoringFunction=utils.maskedNll)
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - ABot Rank'
            viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        # Clean up CUDA cache after evaluation
        torch.cuda.empty_cache()

    # Evaluate Q-Bot ranking
    if 'QBotRank' in params['evalModeList']:
        print("Performing QBotRank evaluation")
        rankMetrics, roundRanks = rankQBot(qBot, dataset, split, verbose=1)
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - QBot Rank'
            viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        for r in range(numRounds + 1):
            for metric, value in roundRanks[r].items():
                plotName = '[Iter %d] %s - QABots Rank Roundwise' % \
                            (iterId, splitName)
                viz.linePlot(r, value, plotName, metric, xlabel='Round')

        # Clean up CUDA cache after evaluation
        torch.cuda.empty_cache()

    # Evaluate Q-Bot and A-Bot joint ranking
    if 'QABotsRank' in params['evalModeList']:
        print("Performing QABotsRank evaluation")
        outputPredFile = "data/visdial/visdial/output_predictions_rollout.h5"
        rankMetrics, roundRanks = rankQABots(
            qBot, aBot, dataset, split, beamSize=params['beamSize'])
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - QABots Rank'
            viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        for r in range(numRounds + 1):
            for metric, value in roundRanks[r].items():
                plotName = '[Iter %d] %s - QBot All Metrics vs Round'%\
                            (iterId, splitName)
                viz.linePlot(r, value, plotName, metric, xlabel='Round')

        # Clean up CUDA cache after evaluation
        torch.cuda.empty_cache()

    # Generate dialogs
    if 'dialog' in params['evalModeList']:
        print("Performing dialog generation...")
        split = 'test'
        outputFolder = "dialog_output/results"
        os.makedirs(outputFolder, exist_ok=True)
        outputPath = os.path.join(outputFolder, "results.json")
        params['cocoDir'] = 'coco/images'
        params['cocoInfo'] = 'coco/image_info_test2014.json'
        dialogDump(
            params,
            dataset,
            split,
            aBot=aBot,
            qBot=qBot,
            beamSize=params['beamSize'],
            savePath=outputPath)

        # Clean up CUDA cache after evaluation
        torch.cuda.empty_cache()

    viz.addText("Evaluation run complete!")

