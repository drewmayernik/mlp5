class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];

        for(let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for(let k = 0; k < this.cols; k++) {
                this.data[i][k] = 0;
            }
        }
    }

    // Return a new matrix given an array
    static fromArray(array) {
        let newMatrix = new Matrix(array.length, 1);
        for(let i = 0; i < array.length; i++) {
            newMatrix.data[i][0] = array[i];
        }
        return newMatrix;
    }

    // Return a new array given a matrix
    static toArray(m) {
        let newArray = [];
        for(let i = 0; i < m.rows; i++) {
            for(let j = 0; j < m.cols; j++) {
                newArray.push(m.data[i][j]);
            }
        }
        return newArray;
    }

    // Return the Matrix product of two matrices
    static multiply(a, b) {
        if(a.cols !== b.rows) {
            console.error('Columns of A must match of rows of B.')
            return undefined;
        }
        let result = new Matrix(a.rows, b.cols);
        for(let i = 0; i < result.rows; i++) {
            for(let j = 0; j < result.cols; j++) {
                // Dot product of values in col
                let sum = 0;
                for(let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Return the Element-Wise Subtraction of two matrices
    static subtract(a, b) {
        if(a.rows == b.rows && a.cols == b.cols) {
            let result = new Matrix(a.rows, a.cols);
            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.data[i][j] = a.data[i][j] - b.data[i][j];
                }
            }
            return result;
        } else {
            console.error("Rows and columns must match.");
        }
    }

    // Return the Element-Wise Addition of two matrices
    static add(a, b) {
        if(a.rows == b.rows && a.cols == b.cols) {
            let result = new Matrix(a.rows, a.cols);
            for(let i = 0; i < result.rows; i++) {
                for(let j = 0; j < result.cols; j++) {
                    result.data[i][j] = a.data[i][j] + b.data[i][j];
                }
            }
            return result;
        } else {
            console.error("Rows and columns must match.");
        }
    }

    // Return the transposed array (switch rows and cols)
    static transpose(m) {
        let result = new Matrix(m.cols, m.rows);
        for(let i = 0; i < m.rows; i++) {
            for(let j = 0; j < m.cols; j++) {
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }

    static map(matrix, fn) {
        let result = new Matrix(matrix.rows, matrix.cols)
        for(let i = 0; i < matrix.rows; i++) {
            for(let j = 0; j < matrix.cols; j++) {
                let val = matrix.data[i][j];
                result.data[i][j] = fn(val);
            }
        }
        return result;
    }

    // Randomize the values within the matrix
    randomize(max, min) {
        if(max) {
            if(min) {
                for (let i = 0; i < this.rows; i++) {
                    for (let j = 0; j < this.cols; j++) {
                        this.data[i][j] = Math.random()*(max-min) + min;
                    }
                }
            } else {
                for (let i = 0; i < this.rows; i++) {
                    for (let j = 0; j < this.cols; j++) {
                        this.data[i][j] = Math.random();
                    }
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] = Math.random();
                }
            }
        }
    }

    // Add a matrix or a number to the elements within the matrix
    add(n) {
        if(n instanceof Matrix) {
            if(this.rows == n.rows && this.cols == n.cols) {
                for (let i = 0; i < this.rows; i++) {
                    for (let j = 0; j < this.cols; j++) {
                        this.data[i][j] += n.data[i][j];
                    }
                }
            } else {
                console.error("Columns and rows must match.");
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += n;
                }
            }
        }
    }

    // Subtract a matrix or a number to the elements within the matrix
    subtract(n) {
        if(n instanceof Matrix) {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] -= n.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] -= n;
                }
            }
        }
    }

    // Multiply a matrix or a number to the elements within the matrix
    multiply(n) {
        if(n instanceof Matrix) {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= n.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= n;
                }
            }
        }
        
    }

    // Transpose the matrix
    transpose() {
        let result = new Matrix(this.cols, this.rows);
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        this.rows = result.rows;
        this.cols = result.cols;
        this.data = result.data;
    }

    // Map a function to all elements within the matrix
    map(fn) {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = fn(val, i, j);
            }
        }
    }

    // Print the Matrix
    print() {
        console.table(this.data);
    }
}