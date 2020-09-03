/** @class Representing a population of NeuralNetworks */
class Population {
    /**
     * Creates an instance of population.
     *
     * @constructor
     * @author: Drew Mayernik
     * @param {number} size The number of members in the population
     * @param {number} mutationRate Rate of mutation in mutated weights and biases (learning rate)
     * @param {number} mutationChance Change of mutation; 0 < n < 1
     * @param {number} input_nodes Number of input nodes within each Neural Network
     * @param {number} hidden_layers Number of hidden layers within each Neural Network
     * @param {number} hidden_nodes Number of hidden nodes within each hidden layer for each NeuralNetwork
     * @param {number} output_nodes Number of output nodes within each output layer for each NeuralNetwork
     */
    constructor(size, mutationRate, mutationChance, input_nodes, hidden_layers, hidden_nodes, output_nodes) {
        this.size = size;
        this.mutationRate = mutationRate;
        this.mutationChance = mutationChance;
        this.generationNumber = 1;

        this.population = []

        for(let i = 0; i < this.size; i++) {
            this.population.push(new NeuralNetwork(input_nodes, hidden_layers, hidden_nodes, output_nodes));
            
        }
    }

    /**
     * Generates the next population
   */
    nextPopulation() {
        let tempPopulation = [];
        for(let i = 0; i < this.size; i++) {
            tempPopulation.push(this.createAChild());
            //tempPopulation.push(this.population[this.getMostFit()].copy());
            //this.population[i] = this.population[this.getMostFit()].copy();
        }
        this.population = [];
        for(let i = 0; i < this.size; i++) {
            this.population.push(tempPopulation[i]);
        }
        this.mutatePopulation();
        this.generationNumber++;
    }

    /**
     * Mutates the population based on the mutation rate and mutation chance
     */
    mutatePopulation() {
        for(let i = 0; i < this.size; i++) {
            this.population[i].mutate(this.mutationRate, this.mutationChance);
        }
    }

    /**
     * Gets the most fit NeuralNetwork in the population
     * @return {number} Returns the index of the population of the Neural Network with the highest fitness value
     */
    getMostFit() {
        let fittest = 0;
        for(let i = 0; i < this.size; i++) {
            if(this.population[i].fitness > this.population[fittest].fitness) {
                fittest = i;
            }
        }
        return fittest;
    }

    /**
     * Gets the most fitness value of the most fit NeuralNetwork
     * @return {number} Returns the fitness value the Neural Network with the highest fitness value
     */
    getBestFitness() {
        return this.population[this.getMostFit()].fitness;
    }

    /**
     * Returns the total fitness of the population as a whole
     * @param {number} limit Ignores this member of the population
     * @return {number} Returns the total fitness
     */
    totalFitness(limit) {
        let total = 0;
        for(let i = 0; i < this.size; i++) {
            if(i != limit) {
                total += this.population[i].fitness;
            }
        }
        return total;
    }

    /**
     * Creates a child based off of two random parents decided through pool selection
     * @return {NeuralNetwork} Returns a new Neural Network made of the crossover of its parents
     */
    createAChild() {
        let mateA = this.pickAMate(-1);
        let mateB = this.pickAMate(-1);
        let a = this.population[mateA];
        let b = this.population[mateB];
        
        let child = NeuralNetwork.crossover(a,b);
        return child;
    }

    /**
     * Picks a parent based off of pool selection
     * @param {Number} limit Define an off limits choice
     * @return {Number} Returns the index of the parent NeuralNetwork
     */
    pickAMate(limit) {

        let total = this.totalFitness(limit);
        
        let tempFitnesses = [];
        for(let i = 0; i < this.size; i++) {
            tempFitnesses.push(this.population[i].fitness / total);
            
        }
        //console.table(tempFitnesses);

        let dart = Math.random();
        let choice = 0;
        while(dart>0) {
            if(tempFitnesses[choice] > 0 && choice != limit) {
                dart -= tempFitnesses[choice];
            }
            choice ++;
        }

        choice--;
        return choice;
    }


    /**
     * Sorts the population by fitness
     */
    sortByFitness() {      
        let sorted = []

        for(let i = 0; i < this.size; i++) {
            let fit = this.getMostFit();
            this.population.pop(fit);
            sorted.push(population[fit].copy());
        }

        this.population = [];
        for(let i = 0; i < this.size; i++) {
            this.population.push(sorted[i]);
        }
    }

    /**
     * Generates the output of a NeuralNetwork
     * @param {Number} id Index of the NeuralNetwork to generate an output for
     * @param {[Number]} inputs Array of inputs to input into the Neural Network
     * @return {[Number]} Returns the outputs from the Neural Network input
     */
    getOutput(id, inputs) {
        return this.population[id].predict(inputs);
    }
    
}
