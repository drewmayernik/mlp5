function sigmoid(x) {
    return 1 / (1 + Math.exp(-1 * x));
}

function dsigmoid(y) {
    return (y * (1-y));
}

class NeuralNetwork {

    constructor(input_nodes, hidden_layers, hidden_nodes, output_nodes) {
        // Number of nodes in each layer
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        // Generate hidden weight matrix and fill it with random values from -1 to 1
        this.weights_ih = new Matrix(hidden_nodes, input_nodes);
        this.weights_ih.randomize(-1,1);

        // Generate hidden bias
        this.bias_ih = new Matrix(hidden_nodes, 1);
        this.bias_ih.randomize(-1,1);

        
        // Generate matrices for the weights and biases of the hidden layers
        this.weights_hh = [];
        this.bias_hh = [];
        for(let i = 0; i < hidden_layers-1; i++) {
            let temp = new Matrix(hidden_nodes, hidden_nodes);
            temp.randomize(-1,1);
            this.weights_hh.push(temp);

            temp = new Matrix(hidden_nodes, 1);
            temp.randomize(-1,1);
            this.bias_hh.push(temp);
        }


        // Generate output weight matrix and fill it with random values from -1 to 1
        this.weights_ho = new Matrix(output_nodes, hidden_nodes);
        this.weights_ho.randomize(-1,1);

        
        // Generate output bias
        this.bias_ho = new Matrix(output_nodes, 1);
        this.bias_ho.randomize(-1,1);

        // Learning rate
        this.learningRate = 0.01;
    }

    setLearningRate(x) {
        this.learningRate = x;
    }


    feedforward(input_array) {
        // Turn array into matrix
        let inputs = Matrix.fromArray(input_array);

        // Find dot product of weights and inputs
        let hidden_output = Matrix.multiply(this.weights_ih, inputs);

        // Add the bias matrix
        hidden_output.add(this.bias_ih);

        // Apply activation function to the hidden outputs
        hidden_output.map(sigmoid);

        for(let i = 0; i < this.weights_hh.length; i++) {
            hidden_output = Matrix.multiply(this.weights_hh[i], hidden_output);
            hidden_output.add(this.bias_hh[i]);
            hidden_output.map(sigmoid);
        }

        // Find dot product of output weights and hidden outputs
        let output = Matrix.multiply(this.weights_ho, hidden_output);

        // Add the output bias matrix
        output.add(this.bias_ho);

        // Apply activation function to the outputs
        output.map(sigmoid);

        // Return the output matrix
        return Matrix.toArray(output);
    }

    train(input_array, target_array) {
        // Turn array into matrix
        let inputs = Matrix.fromArray(input_array);

        // Find dot product of weights and inputs
        let hidden_output = Matrix.multiply(this.weights_ih, inputs);

        // Add the bias matrix
        hidden_output.add(this.bias_ih);

        // Apply activation function to the hidden outputs
        hidden_output.map(sigmoid);
        let save_outputs = [hidden_output];
        //save_outputs.push(hidden_output);
        for(let i = 0; i < this.weights_hh.length; i++) {
            hidden_output = Matrix.multiply(this.weights_hh[i], hidden_output);
            hidden_output.add(this.bias_hh[i]);
            hidden_output.map(sigmoid);
            save_outputs.push(hidden_output);
        }

        // Find dot product of output weights and hidden outputs
        let output = Matrix.multiply(this.weights_ho, hidden_output);

        // Add the output bias matrix
        output.add(this.bias_ho);

        // Apply activation function to the outputs
        output.map(sigmoid);

        let outputs = output;

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
        let hidden_T = Matrix.transpose(save_outputs[this.weights_hh.length]);
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);
        this.weights_ho.add(weights_ho_deltas);
        this.bias_ho.add(gradients);

        
        let lastLayerWeights = this.weights_ho;
        let lastErrors = output_errors;

        for(let i = this.weights_hh.length; i > 0; i--) {
            // Calculate the hidden layer errors
            let llw_t = Matrix.transpose(lastLayerWeights);
            let hidden_errors = Matrix.multiply(llw_t, lastErrors);

            let hidden_gradient = Matrix.map(save_outputs[i], dsigmoid);
            hidden_gradient.multiply(hidden_errors);
            hidden_gradient.multiply(this.learningRate);
            
            let iT = Matrix.transpose(save_outputs[i-1]);
            let weights_hh_deltas = Matrix.multiply(hidden_gradient, iT);

            this.weights_hh[i-1].add(weights_hh_deltas);
            this.bias_hh[i-1].add(hidden_gradient);
            
            lastLayerWeights = this.weights_hh[i-1];
            lastErrors = hidden_errors;
        }


        // Calculate the hidden layer errors
        let who_t = Matrix.transpose(lastLayerWeights);
        let hidden_errors = Matrix.multiply(who_t, lastErrors);

        // Calculate hidden gradient
        let hidden_gradient = Matrix.map(save_outputs[0], dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learningRate);

        // Calculate hidden deltas
        let inputs_T = Matrix.transpose(inputs);
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        this.weights_ih.add(weights_ih_deltas);
        this.bias_ih.add(hidden_gradient);

    }

    print() {
        this.weights_ih.print();
        for(let i = 0; i < this.weights_hh.length; i++) {
            this.weights_hh[i].print();
        }
        this.weights_ho.print();
    }
}