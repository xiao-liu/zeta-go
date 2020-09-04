# -*- coding: utf-8 -*-

import glog as log
import torch
import torch.optim as optim

from compare import Comparator
from config import get_conf
from example import ExamplePool
from network import ZetaGoNetwork


def learning_rate(step, conf):
    for threshold, lr in conf.LR_SCHEDULE:
        if threshold < 0 or step < threshold:
            return lr


def train(model_dir, conf_name, num_workers, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_file is None:
        log.info('start training a new model')
        log.info(f'model_dir={model_dir}')
        log.info(f'conf_name={conf_name}')
        log.info(f'device={device}')
        log.info(f'num_workers={num_workers}')

        conf = get_conf(conf_name)

        iteration = 0
        step = 0

        # randomly initialize a network and let it be the best network
        network = ZetaGoNetwork(conf)
        best_network = ZetaGoNetwork(conf)
        best_network.load_state_dict(network.state_dict())
        network.to(device)
        best_network.to(device)

        # setup the optimizer
        # notice that the L2 regularization is implemented by
        # introducing a weight decay
        optimizer = optim.SGD(
            network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2*conf.L2_REG,
        )

        # create a comparator to compare two networks
        comparator = Comparator(conf)

        # create an example pool and fill it with examples
        log.info('initializing the example pool...')
        example_pool = ExamplePool(conf)
        example_pool.generate_examples(best_network, device, num_workers)
        example_pool.shuffle()
    else:
        log.info(f'resume training from checkpoint {checkpoint_file}')
        log.info(f'model_dir={model_dir}')
        if conf_name != '':
            log.info(f'original configuration overridden by {conf_name}')
        log.info(f'device={device}')
        log.info(f'num_workers={num_workers}')

        # load checkpoint and restore all the necessary states
        checkpoint = torch.load(checkpoint_file)

        conf = get_conf(conf_name) if conf_name != '' else checkpoint['conf']

        iteration = checkpoint['iteration']
        step = checkpoint['step']

        network = ZetaGoNetwork(conf)
        network.load_state_dict(checkpoint['network'])
        best_network = ZetaGoNetwork(conf)
        best_network.load_state_dict(checkpoint['best_network'])
        network.to(device)
        best_network.to(device)

        optimizer = optim.SGD(
            network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2*conf.L2_REG,
        )
        optimizer.load_state_dict(checkpoint['optimizer'])

        comparator = Comparator(conf)

        example_pool = checkpoint['example_pool']

    running_loss = 0.0
    while iteration < conf.NUM_ITERATIONS:
        # train the model
        log.info(f'start iteration {iteration}')
        while example_pool.has_batch():
            # load a batch of examples
            features, pi, z = example_pool.load_batch()
            features = features.to(device)
            pi = pi.to(device)
            z = z.to(device)

            # set network to train mode
            network.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            log_p, v = network(features)
            loss = torch.mean(
                (z - v) ** 2 - torch.sum(pi * log_p, dim=1, keepdim=True))

            # backward
            loss.backward()

            # set learning rate and optimize
            lr = learning_rate(step, conf)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % conf.CHECKPOINT_FREQUENCY == 0:
                log.info(f'[iter={iteration}] checkpoint reached, step={step}')

                # update best_network if the new network is stronger
                # notice that it is necessary to make a copy
                log.info(
                    f'[iter={iteration}] comparing current network with best '
                    f'network...'
                )
                win_rate = comparator.estimate_win_rate(
                    network, best_network, device, num_workers)
                if win_rate > conf.WIN_RATE_MARGIN:
                    log.info(
                        f'[iter={iteration}] best network updated, '
                        f'win_rate={win_rate}'
                    )
                    best_network.load_state_dict(network.state_dict())
                else:
                    log.info(
                        f'[iter={iteration}] best network not updated, '
                        f'win_rate={win_rate}'
                    )

                # save model and print statistics
                running_loss /= conf.CHECKPOINT_FREQUENCY
                torch.save(
                    {
                        'conf': conf,
                        'iteration': iteration,
                        'step': step,
                        'example_pool': example_pool,
                        'best_network': best_network.state_dict(),
                        'network': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    (
                        f'{model_dir}/'
                        f'checkpoint_[iter_{iteration}][step_{step}].pt',
                    ),
                )
                log.info(
                    f'[iter={iteration}] checkpoint saved, '
                    f'running_loss={running_loss}'
                )
                running_loss = 0.0

        log.info(
            f'[iter={iteration}] generating new examples for the next '
            f'iteration...'
        )
        example_pool.generate_examples(best_network, device, num_workers)
        example_pool.shuffle()

        iteration += 1

    # training finished, save the final best network
    model_path = f'{model_dir}/model.pt'
    torch.save(
        {
            'conf': conf,
            'best_network': best_network.state_dict(),
        },
        model_path,
    )
    log.info(f'finished training, model saved to {model_path}')
