import java.util.*;

class NeuralNetwork {
	public float[] output;					
    public float[] hiddenOutput;
    public float [][] weights1;		
    public float [][] weights2;
    
    public int numLayers;
    final int OUTPUT_SIZE = 4;
    final int INPUT_SIZE = 360;
    final int HIDDEN_SIZE = 180;
    
    // NN Constructor
    public NeuralNetwork(int numLayers) {                      
    	this.numLayers = numLayers;   
    	output = new float[OUTPUT_SIZE];
    	if(numLayers == 1)
    		weights1 = new float[INPUT_SIZE][OUTPUT_SIZE];				
    	else {																	// Two layer case.
    		weights1 = new float[INPUT_SIZE][HIDDEN_SIZE];				
            weights2 = new float[HIDDEN_SIZE][OUTPUT_SIZE];
    	}
        createWeights();
    }
    
    // Generate randomized weights.
    private void createWeights() {    
    	Random r1 = new Random();
        Random r = new Random(r1.nextInt(100));
    	if(numLayers == 1) {
    		for(int i = 0; i < INPUT_SIZE; i++) {                              // For every input,
    			for(int j = 0; j < OUTPUT_SIZE; j++) {                         // Distribute its output weights.           
    				weights1[i][j] = ((float)r.nextDouble()-0.5f)*0.001f;       // [inputs][Outputs]
    			}
    		}
    	} 
    	else {				 																				
    		for(int i = 0; i < INPUT_SIZE; i++) {							   // For every input,    
    			for(int j = 0; j < HIDDEN_SIZE; j++) {						   // Distribute its output weights.
    				weights1[i][j] = ((float)r.nextDouble()-0.5f)*0.001f;	   // Distribute: input_weights -> HiddenLayer
    			}
    		}
    		for(int j = 0; j < HIDDEN_SIZE; j++) {							   // For every hidden neuron,
    			for(int k = 0; k < OUTPUT_SIZE; k++) {						   // Distribute its output weights.
    				weights2[j][k] = ((float)r.nextDouble()-0.5f)*0.001f;	   // Distribute: hidden_weights -> OutputLayer
    			}
			}   		
    	}
   }
    
    // Get the activation value. Our sigmoid function.
    private float activation(float x) {
    	double y  = (double)(x);
    	return (float)(1/(1+Math.exp(-y)));
/*        if (x<=-3 && x>=-4) {
          return x*0.02944f+0.13575f;
        }
        else if (1>=Math.abs(x)) {
          return x*0.23106f+0.5f;
        }
        else if (x>=3 && x<=4) {
          return x*0.02944f+0.86425f;
        }
        else {
          x = 1+-x*0.03125f;
          x *= x;
          x *= x;
          x *= x;
          x *= x;
          x *= x;
          return 1/(1+x);
        }*/
    }
    
    // Feed input through network, calculating wSums & output values.
    public float[] fordprop(float[] inputs) {                               // Compute wSum and activation value for each neuron.                               
    	float[] input = inputs;								
        Arrays.fill(output, 0);												// Reset Array to hold current activation values.
    	float[] wSum2 = new float[OUTPUT_SIZE];
    	if(numLayers == 1) {												// One layer case.
    		for(int j = 0; j < OUTPUT_SIZE; j++) {                          // For every output neuron,
    			for(int i = 0; i < INPUT_SIZE; i++) {                       // Calculate the weighted sum of the input weights.	
    				wSum2[j] += weights1[i][j]*input[i];                      
    			}
    			output[j] = activation(wSum2[j]);                           // Calculate that neurons activation value.
    		}
    	}
    	else {         														// Two Layer case.
            twoLayerFordprop(wSum2, input);
    	}
    	return output;
    }
    
    // Two layer Fordprop.
    void twoLayerFordprop(float[] wSum2, float[] input) {					
    	float[] wSum1 = new float[HIDDEN_SIZE];
        hiddenOutput = new float[HIDDEN_SIZE]; 
		for(int j = 0; j < HIDDEN_SIZE; j++) {								// For every hidden neuron,
			for(int i = 0; i < INPUT_SIZE; i++) {							// Calculate the weighted sum of the input weights.		
				wSum1[j] += weights1[i][j]*input[i];						// [] = [inputs][hiddenLayer] * input[inputs].
			}
			hiddenOutput[j] = activation(wSum1[j]);							// Output value of hidden neuron.
		}
		for(int k = 0; k < OUTPUT_SIZE; k++) {								// For every output,				
			for(int j = 0; j < HIDDEN_SIZE; j++) {					        // Calculate the weighted sum of the input weights.
				wSum2[k] += weights2[j][k]*hiddenOutput[j];					// [] = [hiddenLayer][outputLayer] * hiddenOutput[HIDDEN_SIZE].					
			}
			output[k] = activation(wSum2[k]);								// Output value of the output neuron.
		}
    }
}

