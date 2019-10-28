import os
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm


def test(dataloader, model, device, source_target):

	model = model.eval()
	
	with torch.no_grad():

		model = model.to(device)

		target_iter = tqdm(enumerate(dataloader))

		n_total = 0
		n_correct = 0

		for t, batch in target_iter:

			if source_target == 'source':
				x_1, y_task_1, _ = batch
				x = x_1 #torch.cat((x_1, x_2, x_3), dim=0)
				y = y_task_1 #torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
			else:
				x, y, _ = batch

			x = x.to(device)
			y = y.to(device)
				
			out = model.forward(x)	
			class_output = F.softmax(out, dim=1)
			pred = class_output.data.max(1, keepdim=True)[1]
			n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
			n_total += x.size(0)

		acc = n_correct.item() * 1.0 / n_total
		
		print('Current accuracy: {}.'.format(acc))
		
		return acc


