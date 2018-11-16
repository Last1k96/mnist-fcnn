#pragma once
#include "fc_layer.h"

class fc_layer_empty : public fc_layer
{
	void apply_activation_function() override
	{
	}

	void compute_derivatives(const fc_layer& next) override
	{
	}

public:
	explicit fc_layer_empty(vector<double>&& y)
		: fc_layer(0, y.size())
	{
		neurons_ = std::forward<vector<double>>(y);
	}
};
