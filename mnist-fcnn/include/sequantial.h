#pragma once
#include "layer.h"
#include "tanh_layer.h"
#include "softmax_layer.h"

class sequential
{
	layer input_;
	tanh_layer hidden_;
	softmax_layer output_;

	using vec = std::vector<double>;
	using matrix = std::vector<std::vector<double>>;

	void forward_pass(vec x);
	void backward_pass(vec y);
	void adjust_weights(double learning_rate);
	static double cross_entropy(const vec& target, const vec& output);
	static bool is_accurate(const vec& target, const vec& output);
public:

	explicit sequential(size_t input_size, size_t hidden_size, size_t output_size);
	std::tuple<double, double> evaluate(const matrix& x, const matrix& y);
	void fit(const matrix& x_train, const matrix& y_train,
	         const matrix& x_test, const matrix& y_test,
	         int epoch_count = 10, double learning_rate = 0.1, int batch_size = 50);
};
