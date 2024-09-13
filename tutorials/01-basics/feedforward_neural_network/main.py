import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  


# Konfigurasi device untuk menjalankan training di GPU jika tersedia, jika tidak, gunakan CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 gambar yang di-flatten menjadi vektor 1 dimensi
hidden_size1 = 512  # Jumlah neuron di hidden layer pertama
hidden_size2 = 256  # Jumlah neuron di hidden layer kedua
num_classes = 10  # Jumlah kelas (0-9 untuk MNIST)
num_epochs = 15  # Jumlah epoch untuk training
batch_size = 64  # Ukuran mini-batch
learning_rate = 0.001  # Learning rate untuk optimizer

# Dataset MNIST untuk training, gambar akan otomatis di-download jika belum ada
train_dataset = torchvision.datasets.MNIST(root='../../data', #Dataset akan disimpan di folder data
                                           train=True, #Dataset yang ingin diambil adalah untuk training
                                           transform=transforms.ToTensor(), #Gambar diubah menjadi tensor
                                           download=True)

# Dataset MNIST untuk testing
test_dataset = torchvision.datasets.MNIST(root='../../data', #Dataset akan disimpan di folder data
                                          train=False, #Dataset yang ingin diambil adalah untuk testing
                                          transform=transforms.ToTensor()) #Gambar diubah menjadi tensor

# Data loader untuk data training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, # Menggunakan mini-batch ukuran 100
                                           shuffle=True) # Data akan diacak setiap epoch

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False) # Data testing tidak diacak

# Modifikasi arsitektur neural network dengan 2 hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # Fully connected layer pertama
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Fully connected layer kedua
        self.fc3 = nn.Linear(hidden_size2, num_classes)  # Output layer untuk klasifikasi
        self.relu = nn.ReLU()  # Fungsi aktivasi ReLU untuk layer pertama
        self.sigmoid = nn.Sigmoid()  # Fungsi aktivasi Sigmoid untuk layer kedua

    def forward(self, x):
        out = self.fc1(x)  # Input masuk ke layer pertama
        out = self.relu(out)  # Menggunakan fungsi aktivasi ReLU
        out = self.fc2(out)  # Output dari layer pertama masuk ke layer kedua
        out = self.sigmoid(out)  # Menggunakan fungsi aktivasi Sigmoid
        out = self.fc3(out)  # Output dari layer kedua masuk ke layer output
        return out

# Inisialisasi model
model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device) #Mengirim model ke CPU atau GPU

# Definisi loss function (CrossEntropyLoss) dan optimizer (Adam)
criterion = nn.CrossEntropyLoss()  # Menghitung cross entropy loss, cocok untuk masalah klasifikasi
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer Adam untuk update parameter

# Inisialisasi list untuk menyimpan nilai loss setiap iterasi
loss_list = []

# Melakukan training pada model
total_step = len(train_loader)  # Menghitung total jumlah mini-batch di dalam satu epoch
for epoch in range(num_epochs):  # Loop untuk setiap epoch (jumlah pengulangan training dataset)
    for i, (images, labels) in enumerate(train_loader):  # Loop melalui setiap mini-batch dalam satu epoch
        images = images.reshape(-1, 28*28).to(device)  # Mereshape gambar dari [100, 1, 28, 28] menjadi [100, 784] (flattening) dan memindahkannya ke device (GPU/CPU)
        labels = labels.to(device)  # Memindahkan label ke device (GPU/CPU)

        # Forward pass: Mengalirkan input melewati model untuk mendapatkan output prediksi
        outputs = model(images)  # Mendapatkan output prediksi dari model
        loss = criterion(outputs, labels)  # Menghitung loss antara output prediksi dan label sebenarnya

        # Backward pass dan optimasi: Mengupdate bobot model berdasarkan loss
        optimizer.zero_grad()  # Mengatur gradien ke nol sebelum backward pass agar tidak menumpuk
        loss.backward()  # Melakukan backward pass untuk menghitung gradien
        optimizer.step()  # Memperbarui bobot model berdasarkan gradien yang telah dihitung

        # Menyimpan nilai loss dari setiap iterasi untuk keperluan visualisasi nanti
        loss_list.append(loss.item())

        # Menampilkan nilai loss setiap 100 mini-batch untuk monitoring
        if (i+1) % 100 == 0:  # Setiap kelipatan 100 iterasi, cetak informasi loss
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))  # Menampilkan epoch, step, dan nilai loss# Test the model

# Membuat grafik untuk menampilkan nilai loss selama pelatihan
plt.plot(loss_list)  # Menggambar grafik garis menggunakan data dari loss_list
plt.xlabel('Iteration')  # Menambahkan label sumbu x sebagai 'Iteration'
plt.ylabel('Loss')  # Menambahkan label sumbu y sebagai 'Loss'
plt.title('Loss during Training')  # Menambahkan judul grafik sebagai 'Loss during Training'
plt.show()  # Menampilkan grafik yang telah dibuat

# Testing model setelah training selesai
# Pada fase testing, kita tidak perlu menghitung gradient (untuk efisiensi memori)
with torch.no_grad():  # Mematikan perhitungan gradient
    correct = 0  # Menyimpan jumlah prediksi yang benar
    total = 0  # Menyimpan total data yang diuji
    for images, labels in test_loader:  # Loop melalui mini-batch di data testing
        images = images.reshape(-1, 28*28).to(device)  # Mengubah ukuran gambar dan memindahkan ke device
        labels = labels.to(device)
        outputs = model(images)  # Mendapatkan output dari model
        _, predicted = torch.max(outputs.data, 1)  # Mengambil prediksi label (nilai maksimal)
        total += labels.size(0)  # Menambahkan total gambar yang diuji
        correct += (predicted == labels).sum().item()  # Menambahkan jumlah prediksi yang benar

    # Menampilkan akurasi pada 10.000 gambar test
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Menyimpan model yang telah dilatih
torch.save(model.state_dict(), 'model.ckpt')  # Menyimpan state dictionary model ke file
