from trainer import Trainer
import data
from model import get_model
from config import args, config


if __name__ == '__main__':
	loader = data.get_loader()
	model = get_model()

	trainer = Trainer(args, config, model, loader)
	trainer.train()
	trainer.test()