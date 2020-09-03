function sigmoid(x) {
    return 1 / (1 + Math.exp(-1 * x));
}

function dsigmoid(y) {
    return (y * (1-y));
}

function mutate(val, mutationRate, mutationChance) {
    let x = val;
    if(Math.random() > mutationChance) {
        x += ((Math.random()*2)-1) * mutationRate;
    }
    return x;
}

class NeuralNetwork {

    constructor(a, hidden_layers, hidden_nodes, output_nodes) {
        if(a instanceof NeuralNetwork) {
            this.input_nodes = a.input_nodes;
            this.hidden_layers = a.hidden_layers;
            this.hidden_nodes = a.hidden_nodes;
            this.output_nodes = a.output_nodes

            this.weights_ih = a.weights_ih.copy();
            this.bias_ih = a.bias_ih.copy();

            this.weights_hh = []
            this.bias_hh = []
            for(let i = 0; i < this.hidden_layers-1; i++) {
                this.weights_hh.push(a.weights_hh[i].copy());
                this.bias_hh.push(a.bias_hh[i].copy());
            }

            this.weights_ho = a.weights_ho.copy();
            this.bias_ho = a.bias_ho.copy();
            this.fitness = 0;
            
        } else {
            // Number of nodes in each layer
            this.input_nodes = a;
            this.hidden_layers = hidden_layers;
            this.hidden_nodes = hidden_nodes;
            this.output_nodes = output_nodes;

            // Generate hidden weight matrix and fill it with random values from -1 to 1
            this.weights_ih = new Matrix(hidden_nodes, a);
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

           
        }

         // Learning rate
         this.learningRate = 0.01;

         this.fitness = 0;
    }

    static crossover(nn1, nn2) {
        // crossover weights_ih
        let newWeights_ih = new Matrix(nn1.weights_ih.rows, nn1.weights_ih.cols);
        for(let i = 0; i < newWeights_ih.rows; i++) {
            for ( let j = 0; j < newWeights_ih.cols; j++) {
                if(Math.random() > 0.5) {
                    newWeights_ih.data[i][j] = nn1.weights_ih.data[i][j];
                } else {
                    newWeights_ih.data[i][j] = nn2.weights_ih.data[i][j];
                }
            }
        }

        // crossover bias_ih
        let newBias_ih = new Matrix(nn1.bias_ih.rows, nn1.bias_ih.cols);
        for(let i = 0; i < newBias_ih.rows; i++) {
            for ( let j = 0; j < newBias_ih.cols; j++) {
                if(Math.random() > 0.5) {
                    newBias_ih.data[i][j] = nn1.bias_ih.data[i][j];
                } else {
                    newBias_ih.data[i][j] = nn2.bias_ih.data[i][j];
                }
            }
        }

        let newWeights_hh = []
        let newBias_hh = []

        // crossover weights_hh and bias_hh
        for(let k = 0; k < nn1.weights_hh.length; k++) {
            let tempWeights = new Matrix(nn1.weights_hh[k].rows, nn1.weights_hh[k].cols);
            for(let i = 0; i < nn1.weights_hh[k].rows; i++) {
                for ( let j = 0; j < nn1.weights_hh[k].cols; j++) {
                    if(Math.random() > 0.5) {
                        tempWeights.data[i][j] = nn1.weights_hh[k].data[i][j];
                    } else {
                        tempWeights.data[i][j] = nn2.weights_hh[k].data[i][j];
                    }
                }
            }

            newWeights_hh.push(tempWeights.copy());

            let tempBias = new Matrix(nn1.bias_hh[k].rows, nn1.bias_hh[k].cols);
            for(let i = 0; i < nn1.bias_hh[k].rows; i++) {
                for ( let j = 0; j < nn1.bias_hh[k].cols; j++) {
                    if(Math.random() > 0.5) {
                        tempBias.data[i][j] = nn1.bias_hh[k].data[i][j];
                    } else {
                        tempBias.data[i][j] = nn2.bias_hh[k].data[i][j];
                    }
                }
            }

            newBias_hh.push(tempBias.copy());
        }


        // crossover weights_ho
        let newWeights_ho = new Matrix(nn1.weights_ho.rows, nn1.weights_ho.cols);
        for(let i = 0; i < newWeights_ho.rows; i++) {
            for ( let j = 0; j < newWeights_ho.cols; j++) {
                if(Math.random() > 0.5) {
                    newWeights_ho.data[i][j] = nn1.weights_ho.data[i][j];
                } else {
                    newWeights_ho.data[i][j] = nn2.weights_ho.data[i][j];
                }
            }
        }

        // crossover bias_ho
        let newBias_ho = new Matrix(nn1.bias_ho.rows, nn1.bias_ho.cols);
        for(let i = 0; i < newBias_ho.rows; i++) {
            for ( let j = 0; j < newBias_ho.cols; j++) {
                if(Math.random() > 0.5) {
                    newBias_ho.data[i][j] = nn1.bias_ho.data[i][j];
                } else {
                    newBias_ho.data[i][j] = nn2.bias_ho.data[i][j];
                }
            }
        }

        let newNN = new NeuralNetwork(nn1.input_nodes, nn1.hidden_layers, nn1.hidden_nodes, nn1.output_nodes);
        newNN.weights_ih = newWeights_ih.copy();
        newNN.bias_ih = newBias_ih.copy();

        for(let i = 0; i < newNN.weights_hh.length; i++) {
            newNN.weights_hh[i] = newWeights_hh[i].copy();
            newNN.bias_hh[i] = newBias_hh[i].copy();
        }

        newNN.weights_ho = newWeights_ho.copy();
        newNN.bias_ho = newBias_ho.copy();

        return newNN.copy();
    }

    predict(input_array) {
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

    // Neuroevolution functions
    copy() {
        return new NeuralNetwork(this);
    }

    mutate(mutationRate, mutationChance) {
        // mutate weights_ih and bias_ih
        this.weights_ih.map(mutate, mutationRate, mutationChance);
        this.bias_ih.map(mutate, mutationRate, mutationChance);
        // mutate weights_hh and bias_hh
        for(let i = 0; i < this.weights_hh.length; i++) {
            this.weights_hh[i].map(mutate, mutationRate, mutationChance);
            this.bias_hh[i].map(mutate, mutationRate, mutationChance);
        }
        // mutate weights_ho and bias_ho
        this.weights_ho.map(mutate, mutationRate, mutationChance);
        this.bias_ho.map(mutate, mutationRate, mutationChance);
    }

    changeFitness(amount) {
        this.fitness += amount;
    }

    setFitness(amount) {
        this.fitness = amount;
    }
    
    draw(dx, dy, w, h) {
        let xSpacing = w/3;
        let ySpacing = h/(this.input_nodes*2);
        let r = 20;
        let x = dx;
        let y = dy;

        fill(255);
        strokeWeight(1);
        stroke(0);

        let firstY = [];
        let firstX = x + r;

        for(let i = 0; i < this.input_nodes; i++) {
            stroke(0);
            firstY.push(y + r);
            circle(x + r, y + r, r);
            y += ySpacing * 2;
        }

        x += xSpacing;
        y = dy;

        let secondY = [];
        let secondX = x + r;

        for(let i = 0; i < this.hidden_nodes; i++) {
            secondY.push(y + r);
            stroke(0);
            circle(x + r, y + r, r);
            y += ySpacing * 2;
        }

        for(let i = 0; i < this.input_nodes; i++) {
            for(let j = 0; j < this.hidden_nodes; j++) {
                if(this.weights_ih.data[j][i] > 0) stroke(255,0,0);
                else stroke(0, 0, 255);

                strokeWeight(abs(this.weights_ih.data[j][i]));
                line(firstX, firstY[i], secondX, secondY[j]);
            }
        }

        x += xSpacing;
        y = dy;

        let thirdY = [];
        let thirdX = x + r;

        for(let i = 0; i < this.output_nodes; i++) {
            thirdY.push(y + r);
            stroke(0);
            circle(x + r, y + r, r);
            y += ySpacing * 2;
        }

        for(let i = 0; i < this.hidden_nodes; i++) {
            for(let j = 0; j < this.output_nodes; j++) {
                if(this.weights_ho.data[j][i] > 0) stroke(255,0,0);
                else stroke(0, 0, 255);

                strokeWeight(abs(this.weights_ho.data[j][i]));
                line(secondX, secondY[i], thirdX, thirdY[j]);
            }
        }

    }
}