from Data import PennActionData
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.kaming_normal_(m.weight)


        self.fc1 = nn.Linear(26, 64)
        # nn.init.uniform_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(64, 128)
        # nn.init.uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)

        self.fc21 = nn.Linear(128, 2)
        # nn.init.uniform_(self.fc21.weight)
        # nn.init.zeros_(self.fc21.bias)

        self.fc22 = nn.Linear(128, 2)
        # nn.init.uniform_(self.fc22.weight)
        # nn.init.zeros_(self.fc22.bias)

        self.fc3 = nn.Linear(2, 400)
        # nn.init.uniform_(self.fc3.weight)
        # nn.init.zeros_(self.fc3.bias)

        self.fc4 = nn.Linear(400, 26)
        # nn.init.uniform_(self.fc4.weight)
        # nn.init.zeros_(self.fc4.bias)

        # T.nn.init.kaiming_uniform_(self.fc1.weight)
        # T.nn.init.zeros_(self.fc1.bias)
        #
        # T.nn.init.xavier_uniform_(self.fc2.weight)
        # T.nn.init.zeros_(self.fc2.bias)



    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 26))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def scatter_labeled_z(z_batch, label_batch, filename="labeled_z"):
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	colors = ["#800000", "#000075", "#a9a9a9", "#469990", "#bfef45", "#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	for n in range(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[int(label_batch[n])], s=40, marker="o", edgecolors='none')

	classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
	recs = []
	for i in range(0, len(colors)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

	ax = pylab.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
	# pylab.xticks(pylab.arange(-4, 5))
	# pylab.yticks(pylab.arange(-4, 5))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig(filename)

device = torch.device('cuda')

# data_stream = PennActionData(base_dir = '/home/hrishi/1Hrishi/0Thesis/Data/Penn_Action/labels/', scaling = None)
# data_len = data_stream.data_len
# data = data_stream.data
# coord_data = np.concatenate((data['x'], data['y']), axis = 1)
# print(coord_data.shape)
# sequences = data_stream.getStridedSequences(seq_length = seq_length, withVisibility = False)
# np.random.shuffle(sequences)
coord_data = np.load('Raw_Coord_Data_26F.npy')
data_len = len(coord_data)
print(coord_data.shape)
scaler = StandardScaler()
coord_data = scaler.fit_transform(coord_data)
# print(coord_data)

activity_labels = np.load('Raw_activity_labels.npy')

PATH = "Vae.ckpt"
model = VAE()
model.load_state_dict(torch.load(PATH, map_location = 'cpu'))
model.to(device)
# model.train()
model.eval()

''' Evaluation Code '''
with torch.no_grad():
    X_train = torch.from_numpy(coord_data).float().to(device)
    X_train = X_train.view((-1, 26))

    mu, logvar = model.encode(X_train)
    z_batch = model.reparameterize(mu, logvar)

z_batch = z_batch.detach().cpu().numpy()
scatter_labeled_z(z_batch[8700:8840], activity_labels)

''' Training code '''

# params = torch.Tensor(model.parameters())
# #
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.5, patience = 50, verbose = True)
#
# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 26), reduction='elementwise_mean')
#     BCE = F.mse_loss(recon_x, x.view(-1, 26), reduction='elementwise_mean')
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + KLD
#
# num_epochs = 5000
#
# model.train()
# train_losses = []
# #
# for epoch in range(num_epochs):
#     X_train = torch.from_numpy(coord_data).float().to(device)
#     X_train = X_train.view((-1, 26))
#
#     model.zero_grad()
#
#     mu, logvar = model.encode(X_train)
#     z_batch = model.reparameterize(mu, logvar)
#     recon_batch = model.decode(z_batch)
#
#     loss = loss_function(recon_batch, X_train, mu, logvar)
#     # loss.backward()
#
#     # mean_epoch_loss = mean_epoch_train_loss+ loss.item()
#
#     print("Loss over {} data points: {} @ epoch #{}/{}".format(data_len, loss, epoch, num_epochs))
#     train_losses.append(loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     scheduler.step(train_losses[-1])
#
# fig = plt.figure()
# plt.plot(train_losses, 'k')
# plt.show()
#
# checkpoint_file = "Vae.ckpt"
# torch.save(model.state_dict(), checkpoint_file)
