using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNumbers
{
    internal class Layer
    {
        public int numInputs;
        public int numOutputs;
        public float[] bias;
        public float[][] weightValues;
        float[,] weightGradients;

        public Layer(int numInputs, int numOutputs)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            weightValues = new float[numOutputs][];
            for(int i = 0; i < numOutputs; i++)
            {
                weightValues[i] = new float[numInputs];
            }
            weightGradients = new float[numInputs, numOutputs];
            bias = new float[numOutputs];
            InitValues();
        }

        // Polymorphism - when loading, set values from variables instead of InitValues
        public Layer(int numInputs, int numOutputs, float[][] weights, float[] bias)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.weightValues = weights;
            this.bias = bias;
        }

        //Initialize weight values
        private void InitValues()
        {
            Random rand = new Random();
            for (int i = 0; i < numOutputs; i++)
            {
                bias[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
                for (int j = 0; j < numInputs; j++)
                {
                    weightValues[i][j] = (float)(rand.NextDouble() * 2.0 - 1.0);
                }
            }
        }

        // Helper sigmoid function
        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value * -1);
            return 1 / (1.0f + k);
        }

        // Prediction - returns the result to forward to the next layer
        public (float[], float[]) CalculateResult(float[] inputs)
        {
            float[] result = new float[numOutputs];
            float[] rawResult = new float[numOutputs];

            for (int j = 0; j < numOutputs; j++)
            {
                float sum = 0.0f;
                for (int i = 0; i < numInputs; i++)
                {
                    sum += inputs[i] * weightValues[j][i] + bias[j];
                }
                result[j] = Sigmoid(sum);
                rawResult[j] = sum;
            }

            return (result, rawResult);
        }

        // Back propagation
        /*public float[] Backpropagate(float[] inputs, float[] errors, float learningRate)
        {
            float[] inputErrors = new float[numInputs];

            // Compute gradients
            for (int j = 0; j < numOutputs; j++)
            {
                for (int i = 0; i < numInputs; i++)
                {
                    float output = Sigmoid(inputs[i] * weightValues[j, i] + bias[j]);
                    float gradient = errors[j] * SigmoidDerivative(output);
                    weightGradients[j, i] = gradient;
                    inputErrors[i] += gradient * weightValues[j, i];
                }
            }

            // Update weights and bias
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    weightValues[i, j] -= learningRate * weightGradients[i, j];
                    bias[j] -= learningRate * errors[i];
                }
            }

            return inputErrors;
        }
*/
        // Update weights and bias
        private void UpdateWeightsAndBias()
        {

        }

        // Save layer
        public void Save(StreamWriter sw)
        {
            sw.WriteLine(numInputs);
            sw.WriteLine(numOutputs);
            for (int i = 0; i < numOutputs; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    sw.WriteLine(weightValues[i][j]);
                    if(i == 0)
                    {
                        sw.WriteLine(bias[i]);
                    }
                }
            }
        }

        // Load layer
        public static Layer Load(StreamReader sr)
        {
            int numInputs = int.Parse(sr.ReadLine());
            int numOutputs = int.Parse(sr.ReadLine());
            float[] bias = new float[numOutputs];
            float[][] weights = new float[numOutputs][];
            for (int i = 0; i < numOutputs; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    weights[i][j] = float.Parse(sr.ReadLine());
                    if(i == 0)
                    {
                        bias[i] = float.Parse(sr.ReadLine());
                    }
                }
            }
            return new Layer(numInputs, numOutputs, weights, bias);
        }
    }
}
