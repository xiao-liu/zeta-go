# -*- coding: utf-8 -*-

import glog as log
import torch

from state import STAGE_GENERATING, STAGE_TRAINING, State


def learning_rate(step, conf):
    for threshold, lr in conf.LR_SCHEDULE:
        if threshold < 0 or step < threshold:
            return lr


def train(model_dir, conf_name, num_workers, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = State()

    if checkpoint_file is None:
        log.info('start training a new model')
        log.info(f'model_dir={model_dir}')
        log.info(f'conf_name={conf_name}')
        log.info(f'device={device}')
        log.info(f'num_workers={num_workers}')

        # create a new state
        state.create(model_dir, conf_name, device)
        state.save()
        state.example_pool.save(state.iteration)
    else:
        log.info(f'resume training from checkpoint {checkpoint_file}')
        log.info(f'model_dir={model_dir}')
        if conf_name != '':
            log.info(f'original configuration overridden by {conf_name}')
        log.info(f'device={device}')
        log.info(f'num_workers={num_workers}')

        # load state from checkpoint file
        state.load(checkpoint_file, model_dir, conf_name, device)

    running_loss = 0.0
    while state.iteration < state.conf.NUM_ITERATIONS:
        log.info(f'start iteration {state.iteration}')

        if state.stage == STAGE_GENERATING:
            state.example_pool.generate_examples(
                state.iteration, state.best_network, device, num_workers)

            state.example_pool.shuffle()

            state.stage = STAGE_TRAINING
            state.save()

        # train the model
        while state.example_pool.has_batch():
            # load a batch of examples
            features, pi, z = state.example_pool.load_batch()
            features = features.to(device)
            pi = pi.to(device)
            z = z.to(device)

            # set network to train mode
            state.network.train()

            # zero the parameter gradients
            state.optimizer.zero_grad()

            # forward
            log_p, v = state.network(features)
            loss = torch.mean(
                (z - v) ** 2 - torch.sum(pi * log_p, dim=1, keepdim=True))

            # backward
            loss.backward()

            # set learning rate and optimize
            lr = learning_rate(state.step, state.conf)
            for group in state.optimizer.param_groups:
                group['lr'] = lr
            state.optimizer.step()

            running_loss += loss.item()
            state.step += 1

            if state.step % state.conf.CHECKPOINT_FREQUENCY == 0:
                log.info(
                    f'[iter={state.iteration}] checkpoint reached, '
                    f'step={state.step}'
                )

                # update best_network if the new network is stronger
                # notice that it is necessary to make a copy
                log.info(
                    f'[iter={state.iteration}] comparing current network with '
                    f'best network...'
                )
                win_rate = state.comparator.estimate_win_rate(
                    state.network, state.best_network, device, num_workers)
                if win_rate > state.conf.WIN_RATE_MARGIN:
                    log.info(
                        f'[iter={state.iteration}] best network updated, '
                        f'win_rate={win_rate}'
                    )
                    state.best_network.load_state_dict(
                        state.network.state_dict())
                else:
                    log.info(
                        f'[iter={state.iteration}] best network not updated, '
                        f'win_rate={win_rate}'
                    )

                # save model and print statistics
                running_loss /= state.conf.CHECKPOINT_FREQUENCY
                state.save()
                log.info(
                    f'[iter={state.iteration}] checkpoint saved, '
                    f'running_loss={running_loss}'
                )
                running_loss = 0.0

        state.iteration += 1
        state.example_pool.prepare_generation()
        state.stage = STAGE_GENERATING
        state.save()
        state.example_pool.save(state.iteration)

    # training finished, save the final best network
    model_path = f'{model_dir}/model.pt'
    torch.save(
        {
            'conf': state.conf,
            'best_network': state.best_network.state_dict(),
        },
        model_path,
    )
    log.info(f'finished training, model saved to {model_path}')
