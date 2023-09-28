import argparse
from src.drl.train_async import *
from src.gan.adversarial_train import *
from src.drl.train_sinproc import set_SAC_parser, train_SAC
from src.drl.egsac.train_egsac import set_EGSAC_parser, train_EGSAC
from src.drl.sunrise.train_sunrise import train_SUNRISE, set_SUNRISE_args
from src.drl.dvd import set_DvDSAC_parser, train_DvDSAC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gan = subparsers.add_parser('gan', help='Train GAN')
    set_GAN_parser(parser_gan)
    parser_gan.set_defaults(entry=train_GAN)

    parser_sac = subparsers.add_parser('sac', help='Train SAC')
    set_SAC_parser(parser_sac)
    parser_sac.set_defaults(entry=train_SAC)

    parser_asyncsac = subparsers.add_parser('asyncsac', help='Train Asynchronous SAC')
    set_AsyncSAC_parser(parser_asyncsac)
    parser_asyncsac.set_defaults(entry=train_AsyncSAC)

    parser_egsac = subparsers.add_parser('egsac', help='Train Episodic Generative SAC')
    set_EGSAC_parser(parser_egsac)
    parser_egsac.set_defaults(entry=train_EGSAC)

    parser_ncesac = subparsers.add_parser('ncesac', help='Train Negatively Correlated Ensemble SAC')
    set_NCESAC_parser(parser_ncesac)
    parser_ncesac.set_defaults(entry=train_NCESAC)

    parser_pmoesac = subparsers.add_parser('pmoe', help='Train PMOE')
    set_PMOESAC_parser(parser_pmoesac)
    parser_pmoesac.set_defaults(entry=train_PMOESAC)

    parser_sunrise = subparsers.add_parser('sunrise', help='Train SUNRISE')
    set_SUNRISE_args(parser_sunrise)
    parser_sunrise.set_defaults(entry=train_SUNRISE)

    parser_dvd = subparsers.add_parser('dvd', help='Train DvD')
    set_DvDSAC_parser(parser_dvd)
    parser_dvd.set_defaults(entry=train_DvDSAC)

    args = parser.parse_args()

    entry = args.entry
    entry(args)
