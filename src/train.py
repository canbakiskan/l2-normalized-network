import time
import os
from tqdm import tqdm
import logging
import torch
from .train_test_functions import train, test, train_activation_hoyer
from .models.layer_output_extractor import LayerOutputExtractor_wrapper
from .parameters import get_arguments
from .utils.read_datasets import cifar10
from .utils.get_optimizer_scheduler import get_optimizer_scheduler
from .utils.device import determine_device
from .utils.namers import (
    classifier_ckpt_namer,
    classifier_log_namer,
)
from .utils.get_modules import create_classifier
from .utils.logger import logger_setup
import sys

logger = logging.getLogger(__name__)


def check_recompute():
    print(
        "Checkpoint already exists. Do you want to retrain? [y/(n)]", end=" ")
    response = input()
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    if response != "y":
        exit()


def apply_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(args, classifier):

    if not os.path.exists(os.path.dirname(classifier_ckpt_namer(args))):
        os.makedirs(os.path.dirname(classifier_ckpt_namer(args)))

    classifier_filepath = classifier_ckpt_namer(args)
    torch.save(classifier.state_dict(), classifier_filepath)

    logger.info(f"Saved to {classifier_filepath}")


def main():

    args = get_arguments()

    if os.path.exists(classifier_ckpt_namer(args)):
        check_recompute()

    logger_setup(classifier_log_namer(args))
    logger.info(args)
    logger.info("\n")

    device = determine_device(args)
    apply_seed(args.seed)

    train_loader, test_loader = cifar10(args)

    model = create_classifier(args)

    if args.hoyer_activation_lambda != 0.0:
        model = LayerOutputExtractor_wrapper(
            model, args.neural_net.hoyer_layers)

    model.train()

    # if device.type == "cuda":
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    optimizer, scheduler = get_optimizer_scheduler(args, model)

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")

    for epoch in tqdm(range(args.neural_net.epochs)):
        start_time = time.time()
        if args.hoyer_activation_lambda == 0.0:
            # pass
            train_loss, train_acc = train(
                model, train_loader, optimizer, scheduler)
        else:
            train_loss, train_acc = train_activation_hoyer(
                model, train_loader, optimizer, scheduler, args.hoyer_activation_lambda)

        test_loss, test_acc = test(model, test_loader)

        end_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info(
            f"{epoch+1} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}"
        )
        logger.info(
            f"Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

    if args.neural_net.save_checkpoint:
        save_model(args, model)


if __name__ == "__main__":
    main()
