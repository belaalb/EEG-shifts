import os
import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import Loader_validation
import models as models

import sys
sys.path.append('../')
import utils

class OffTest(object):

	def __init__(self, checkpoint, loader, cuda):

		self.cp_task = checkpoint
		self.dataloader = loader
		self.cuda = cuda
		self.feature_extractor = models.AlexNet(num_classes = 7, baseline = False)
		self.task_classifier = models.task_classifier()

	def load_checkpoint(self, epoch=None):
		self.cp_task = self.cp_task.format(epoch) if epoch else self.cp_task
		
		if os.path.isfile(self.cp_task):
			ckpt = torch.load(self.cp_task)
			not_loaded = self.feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
			not_loaded = self.task_classifier.load_state_dict(ckpt['task_classifier_state'])
		else:
			print('No checkpoint found at: {}'.format(self.cp_task))


	def test(self):

		feature_extractor = self.feature_extractor.eval()
		task_classifier = self.task_classifier.eval()
		
		with torch.no_grad():

			if self.cuda:
				self.feature_extractor = self.feature_extractor.cuda()
				self.task_classifier = self.task_classifier.cuda()

			target_iter = tqdm(enumerate(self.dataloader))

			n_total = 0
			n_correct = 0
			predictions_domain = []
			labels_domain = []
			for t, batch in target_iter:
				x, y, _ = batch
				if self.cuda:
					x = x.cuda()
					y = y.cuda()
				
				features = self.feature_extractor.forward(x)
				task_out = self.task_classifier.forward(features)
				class_output = F.softmax(task_out, dim=1)
				pred_task = class_output.data.max(1, keepdim=True)[1]
				n_correct += pred_task.eq(y.data.view_as(pred_task)).cpu().sum()
				n_total += x.size(0)
				
			acc = n_correct.item() * 1.0 / n_total			
			
			return acc

if __name__ == '__main__':
	domains = ['sketch', 'cartoon', 'photo', 'artpainting']
	loaders_dict = {}
	cuda = True
	img_transform = transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	results_dict = {}
	results_dict['Alexnet'] = []
	matrix_dict = {}
	matrix_dict['Alexnet'] = []
	models = ['Alexnet']

	for domain in domains:
		data_path =  './prepared_data/test_'+ domain +'.hdf'
		dataset = Loader_validation(hdf_path=data_path, transform=img_transform)
		loaders_dict[domain] = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=4)

	for model in models:
		cp_path = './'+model
		disparity_matrix = np.zeros([len(domains), len(domains)])

		for d1, domain in enumerate(domains):
			cp_name = os.glob(os.join(cp_path+domain)+'*.pt')
			test_object = OffTest(cp_path, cp_name, loader, cuda)
			test_object.load_checkpoint()
			for d2, domain in enumerate(domains):
				acc = test_object.test()
			disparity_matrix[d1, d2] = 1-acc	
				
			for d1 in range(len(domains)):
				for d2 in range(len(domains)):
					disparity_matrix[d1, d2] = min(min(disparity_matrix[d1, d2], 1-disparity_matrix[d1, d2]), min(disparity_matrix[d2, d1], 1-disparity_matrix[d2, d1]))
					disparity_matrix[d2, d1] = disparity_matrix[d1, d2] 
			
			matrix_dict[model].append(disparity_matrix)

			cond_shift = utils.disparity_normalized_frob_norm(disparity_matrix)
			results_dict[model].append(cond_shift)
			print('Norm of disparity matrix:', cond_shift)

			if len(matrix_dict[model]) <= 1:
				fmt = lambda x, pos: '{:.3f}'.format(x)
				disparity_hmap = sns.heatmap(disparity_matrix, annot=True, fmt='.3f', cmap="RdPu", cbar_kws={'format': FuncFormatter(fmt)})
				plt.savefig('disparity_hmap_pink.png')
				plt.show()

	pd.DataFrame.from_dict(results_dict).to_csv('./Results/all_normalizations_condshift.csv', index=False)

	# Saving disparity matrices 
	for key in matrix_dict.keys():
		values = matrix_dict[key]
		matrix_dict[key] = pd.DataFrame(values, columns=['sketch', 'cartoon', 'photo', 'artpainting']).assign(Model=key)

	#df_subs = pd.concat([matrix_dict['Alexnet'], matrix_dict['Resnet']])
	df_subs = matrix_dict['Alexnet']
	df_subs.to_csv('./Results/average_disparity_matrix.csv', index=False)	
				

