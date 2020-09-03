class Population {
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

    mutatePopulation() {
        for(let i = 0; i < this.size; i++) {
            this.population[i].mutate(this.mutationRate, this.mutationChance);
        }
    }

    getMostFit() {
        let fittest = 0;
        for(let i = 0; i < this.size; i++) {
            if(this.population[i].fitness > this.population[fittest].fitness) {
                fittest = i;
            }
        }
        return fittest;
    }

    getBestFitness() {
        return this.population[this.getMostFit()].fitness;
    }

    totalFitness(limit) {
        let total = 0;
        for(let i = 0; i < this.size; i++) {
            if(i != limit) {
                total += this.population[i].fitness;
            }
        }
        return total;
    }

    createAChild() {
        let mateA = this.pickAMate(-1);
        let mateB = this.pickAMate(-1);
        let a = this.population[mateA];
        let b = this.population[mateB];
        
        let child = NeuralNetwork.crossover(a,b);
        return child;
    }

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



    sortByFitness() {
        let sorted = []

        for(let i = 0; i < this.size; i++) {
            let fit = this.getMostFit();
            this.population.pop(fit);
            sorted.push(population[fit].copy());
        }

        this.population = sorted.slice();
    }

    getOutput(id, inputs) {
        return this.population[id].predict(inputs);
    }
    
}