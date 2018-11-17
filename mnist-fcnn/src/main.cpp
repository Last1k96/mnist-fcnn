#include "utils.h"
#include "mnist_reader.h"
#include "sequantial.h"
#include <filesystem>
#include <iostream>
#include <ctime>

using std::cout;
namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		cout << "Usage:\n";
		cout << fs::path(argv[0]).filename().string() << ' ';
		cout << "[MNIST_path] [epoch_count] [learning_rate] [hidden_layer_size] [batch_size] [train_count]\n\n";
		cout << "Using default values:\n";
	}

	// default values
	auto MNIST_path = fs::path("mnist");
	auto epoch_count = 10;
	auto learning_rate = 0.2;
	auto hidden_size = 100u;
	auto batch_size = 100;
	auto train_count = 60000;

	// reading params
	for (auto i = 1; i < argc; i++)
	{
		switch (i)
		{
		case 1: MNIST_path = fs::path(argv[i]);
			break;
		case 2: epoch_count = std::stoi(argv[i]);
			break;
		case 3: learning_rate = std::stod(argv[i]);
			break;
		case 4: hidden_size = std::stoul(argv[i]);
			break;
		case 5: batch_size = std::stoi(argv[i]);
			break;
		case 6: train_count = std::stoi(argv[i]);
			break;
		default: break;
		}
	}

	cout << "MNIST path: " << MNIST_path.string() << '\n';
	cout << "Epoch count: " << epoch_count << '\n';
	cout << "Learning rate: " << learning_rate << '\n';
	cout << "Hidden layer size: " << hidden_size << '\n';
	cout << "Batch size: " << batch_size << '\n';
	cout << "Training images count: " << train_count << '\n';

	std::ofstream myfile;
	myfile.open("test.txt", std::ios_base::app);
	myfile << "\nEpoch count: " << epoch_count << '\n';
	myfile << "Learning rate: " << learning_rate << '\n';
	myfile << "Hidden layer size: " << hidden_size << '\n';
	myfile << "Batch size: " << batch_size << '\n';
	myfile.close();

	cout << std::fixed << std::setprecision(4);
	srand(static_cast<unsigned>(time(nullptr)));
	try
	{
		cout << "Loading data...\n";

		auto [train_images, train_labels] = mnist_reader::read(
			(MNIST_path / "train-images.idx3-ubyte").string(),
			(MNIST_path / "train-labels.idx1-ubyte").string()
		);

		auto [test_images, test_labels] = mnist_reader::read(
			(MNIST_path / "t10k-images.idx3-ubyte").string(),
			(MNIST_path / "t10k-labels.idx1-ubyte").string()
		);

		cout << "Loaded.\n\n";

		const auto [random_images, random_labels] = utils::random_subset(train_images, train_labels, train_count);

		const auto x_train = utils::normalize_image_set(random_images);
		const auto y_train = utils::to_categorical(random_labels);

		const auto x_test = utils::normalize_image_set(test_images);
		const auto y_test = utils::to_categorical(test_labels);

		const auto input_size = x_train[0].size();
		const auto num_classes = y_train[0].size();

		auto model = sequential{input_size, hidden_size, num_classes};
		model.fit(x_train, y_train, x_test, y_test, epoch_count, learning_rate, batch_size);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}
	return 0;
}
