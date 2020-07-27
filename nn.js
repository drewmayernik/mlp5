function sigmoid(x) {
    return 1 / (1 + Math.exp(-1 * x));
}

function dsigmoid(y) {
    return (y * (1-y));
}

class NeuralNetwork {

    constructor(input_nodes, hidden_nodes, output_nodes, learningRate) {
        // Number of nodes in each layer
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        // Generate hidden weight matrix and fill it with random values from -1 to 1
        this.weights_ih = new Matrix(hidden_nodes, input_nodes);
        this.weights_ih.randomize(-1,1);

        // Generate output weight matrix and fill it with random values from -1 to 1
        this.weights_ho = new Matrix(output_nodes, hidden_nodes);
        this.weights_ho.randomize(-1,1);

        // Generate hidden bias
        this.bias_h = new Matrix(hidden_nodes, 1);
        this.bias_h.randomize(-1,1);

        // Generate output bias
        this.bias_o = new Matrix(output_nodes, 1);
        this.bias_o.randomize(-1,1);

        // Learning rate
        this.learningRate = learningRate;
    }

    feedforward(input_array) {
        // Turn array into matrix
        let inputs = Matrix.fromArray(input_array);

        // Find dot product of weights and inputs
        let hidden = Matrix.multiply(this.weights_ih, inputs);

        // Add the bias matrix
        hidden.add(this.bias_h);

        // Apply activation function to the hidden outputs
        hidden.map(sigmoid);

        // Find dot product of output weights and hidden outputs
        let output = Matrix.multiply(this.weights_ho, hidden);

        // Add the output bias matrix
        output.add(this.bias_o);

        // Apply activation function to the outputs
        output.map(sigmoid);

        // Return the output matrix
        return Matrix.toArray(output);
    }

    train(input_array, target_array) {
        // Turn array into matrix
        let inputs = Matrix.fromArray(input_array);
        // Find dot product of weights and inputs
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        // Add the bias matrix
        hidden.add(this.bias_h);
        // Apply activation function to the hidden outputs
        hidden.map(sigmoid);

        // Find dot product of output weights and hidden outputs
        let outputs = Matrix.multiply(this.weights_ho, hidden);
        // Add the output bias matrix
        outputs.add(this.bias_o);
        // Apply activation function to the outputs
        outputs.map(sigmoid);


        // Convert target array to matrix
        let targets = Matrix.fromArray(target_array);

        // Calculate the error.  Error = targets - outputs
        let output_errors = Matrix.subtract(targets, outputs);
        // Calculate the output gradient
        // learning rate * Error * dSigmoid(outputs) * hidden output (transposed)
        let gradients = Matrix.map(outputs, dsigmoid);
        gradients.multiply(output_errors);
        gradients.multiply(this.learningRate);
        
        // Calculate the output deltas
        let hidden_T = Matrix.transpose(hidden);
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);
        this.weights_ho.add(weights_ho_deltas);
        this.bias_o.add(gradients);

        // Calculate the hidden layer errors
        let who_t = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // Calculate hidden gradient
        let hidden_gradient = Matrix.map(hidden, dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learningRate);

        // Calculate hidden deltas
        let inputs_T = Matrix.transpose(inputs);
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);

    }
}